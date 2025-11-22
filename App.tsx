import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Upload, 
  ZoomIn, 
  ZoomOut, 
  ChevronLeft, 
  ChevronRight, 
  MessageSquarePlus, 
  Crop, 
  X, 
  BrainCircuit,
  Loader2,
  FileText,
  Trash2,
  Settings,
  Download,
  Edit2,
  Save as SaveIcon,
  Sparkles,
  MousePointer2,
  PanelRight,
  GripVertical
} from 'lucide-react';

// --- Types ---

type AnnotationType = 'text_highlight' | 'region_crop' | 'page_note';
type LLMProvider = 'gemini' | 'openai' | 'anthropic';
type InteractionMode = 'read' | 'ask' | 'crop';

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Annotation {
  id: string;
  pageNumber: number;
  type: AnnotationType;
  selectedText?: string;
  rect: Rect;
  rects?: Rect[]; 
  imageBase64?: string;
  userPrompt: string;
  llmResponse: string;
  manualNote: string;
  timestamp: number;
  providerUsed?: LLMProvider;
}

// --- LLM Services ---

const callGemini = async (apiKey: string, systemPrompt: string, userPrompt: string, contextData: { text?: string; image?: string }) => {
  const baseUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`;
  const parts: any[] = [];
  
  if (contextData.text) parts.push({ text: `CONTEXT TEXT:\n"${contextData.text}"\n\n` });
  if (contextData.image) {
    const base64Data = contextData.image.split(',')[1];
    parts.push({ inlineData: { mimeType: "image/png", data: base64Data } });
  }
  parts.push({ text: `SYSTEM: ${systemPrompt}\nUSER: ${userPrompt}` });

  const response = await fetch(baseUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: [{ parts }] })
  });

  if (!response.ok) throw new Error((await response.json()).error?.message || 'Gemini Error');
  const data = await response.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response.";
};

const callOpenAI = async (apiKey: string, systemPrompt: string, userPrompt: string, contextData: { text?: string; image?: string }) => {
  const content: any[] = [{ type: "text", text: userPrompt }];
  if (contextData.text) content.unshift({ type: "text", text: `CONTEXT FROM PDF:\n"${contextData.text}"\n\n` });
  if (contextData.image) content.push({ type: "image_url", image_url: { url: contextData.image } });

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
    body: JSON.stringify({
      model: "gpt-4o",
      messages: [{ role: "system", content: systemPrompt }, { role: "user", content: content }],
      max_tokens: 1000
    })
  });

  if (!response.ok) throw new Error((await response.json()).error?.message || 'OpenAI Error');
  const data = await response.json();
  return data.choices?.[0]?.message?.content || "No response.";
};

const callAnthropic = async (apiKey: string, systemPrompt: string, userPrompt: string, contextData: { text?: string; image?: string }) => {
  const content: any[] = [];
  if (contextData.text) content.push({ type: "text", text: `CONTEXT FROM PDF:\n"${contextData.text}"\n\n` });
  if (contextData.image) {
    const base64Data = contextData.image.split(',')[1];
    content.push({ type: "image", source: { type: "base64", media_type: "image/png", data: base64Data } });
  }
  content.push({ type: "text", text: userPrompt });

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
      'dangerously-allow-browser': 'true'
    },
    body: JSON.stringify({
      model: "claude-3-5-sonnet-20240620",
      max_tokens: 1000,
      system: systemPrompt,
      messages: [{ role: "user", content }]
    })
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error?.message || 'Anthropic Error (Check CORS/Proxy)');
  }
  const data = await response.json();
  return data.content?.[0]?.text || "No response.";
};

const downloadBlob = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// --- Components ---

const Modal = ({ isOpen, onClose, title, children }: any) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-lg flex flex-col max-h-[90vh]">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="font-semibold text-lg text-gray-800">{title}</h3>
          <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded-full"><X size={20} /></button>
        </div>
        <div className="p-4 overflow-y-auto flex-1">
          {children}
        </div>
      </div>
    </div>
  );
};

// --- Sub-Component: PDFPage ---
// Handles rendering, cancellation, and proper sizing
const PDFPage = ({ 
  pageNum, 
  pdfDoc, 
  pdfLib,
  scale, 
  annotations, 
  interactionMode, 
  onMouseDown, 
  onMouseMove, 
  onMouseUp, 
  onTextSelection,
  selectionRect,
  isInteractingWithThisPage,
  setPageRef
}: any) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);
  const pageRef = useRef<HTMLDivElement>(null);
  const renderTaskRef = useRef<any>(null); // Track the active render task
  const [isRendering, setIsRendering] = useState(true);

  useEffect(() => {
    if (setPageRef && pageRef.current) {
      setPageRef(pageNum, pageRef.current);
    }
  }, [pageNum, setPageRef]);

  useEffect(() => {
    const render = async () => {
      if (!pdfDoc || !canvasRef.current || !textLayerRef.current || !pdfLib) return;
      
      // Cancel any existing render task for this page
      if (renderTaskRef.current) {
        try {
          await renderTaskRef.current.cancel();
        } catch (e) {
          // Cancellation expected
        }
      }

      setIsRendering(true);
      
      try {
        const page = await pdfDoc.getPage(pageNum);
        const viewport = page.getViewport({ scale });
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        // Set dimensions
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const textLayerDiv = textLayerRef.current;
        textLayerDiv.innerHTML = '';
        textLayerDiv.style.height = `${viewport.height}px`;
        textLayerDiv.style.width = `${viewport.width}px`;
        textLayerDiv.style.setProperty('--scale-factor', `${scale}`);

        const renderContext = { canvasContext: context, viewport };
        
        // Store the render task so we can cancel it if needed
        const renderTask = page.render(renderContext);
        renderTaskRef.current = renderTask;
        
        await renderTask.promise;
        renderTaskRef.current = null; // Clear on success

        // Only render text layer if canvas finished
        const textContent = await page.getTextContent();
        await pdfLib.renderTextLayer({ 
          textContent: textContent, 
          container: textLayerDiv, 
          viewport, 
          textDivs: [] 
        }).promise;

      } catch (err: any) {
        if (err?.name !== 'RenderingCancelledException') {
          console.error(`Error rendering page ${pageNum}`, err);
        }
      } finally {
        setIsRendering(false);
      }
    };

    render();

    // Cleanup function to cancel render on unmount/update
    return () => {
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
      }
    };
  }, [pdfDoc, pageNum, scale, pdfLib]);

  return (
    <div 
      ref={pageRef}
      className="relative bg-white shadow-md mb-6 shrink-0" 
      style={{ minWidth: '200px', minHeight: '200px' }}
      onMouseDown={(e) => onMouseDown(e, pageNum)}
      onMouseMove={(e) => onMouseMove(e, pageNum)}
      onMouseUp={(e) => onMouseUp(e, pageNum)}
    >
      <canvas ref={canvasRef} className="block" />
      {isRendering && <div className="absolute inset-0 bg-white flex items-center justify-center z-10"><Loader2 className="animate-spin text-indigo-500" /></div>}
      
      {/* Text Layer for Selection */}
      <div 
        ref={textLayerRef} 
        className="textLayer" 
        onMouseUp={(e) => onTextSelection(e, pageNum, pageRef.current)}
        style={{ pointerEvents: interactionMode === 'crop' ? 'none' : 'auto' }}
      />

      {/* Drag Selection Rect */}
      {isInteractingWithThisPage && selectionRect && (
        <div 
          className="absolute border-2 border-indigo-500 bg-indigo-500/20 pointer-events-none z-20"
          style={{ left: selectionRect.x, top: selectionRect.y, width: selectionRect.w, height: selectionRect.h }}
        />
      )}

      {/* Persistent Highlights */}
      {annotations.map((ann: Annotation) => {
         if (ann.type === 'text_highlight' && ann.rects && ann.rects.length > 0) {
          return (
            <React.Fragment key={ann.id}>
              {ann.rects.map((r, i) => (
                <div
                  key={`${ann.id}-${i}`}
                  className="absolute bg-yellow-500/20 hover:bg-yellow-500/30 cursor-pointer transition z-10 mix-blend-multiply"
                  style={{
                    left: `${r.x * 100}%`,
                    top: `${r.y * 100}%`,
                    width: `${r.w * 100}%`,
                    height: `${r.h * 100}%`
                  }}
                  title={ann.userPrompt}
                />
              ))}
            </React.Fragment>
          );
        }
        return (
          <div
            key={ann.id}
            className={`absolute border-2 cursor-pointer transition z-10 ${ann.type === 'region_crop' ? 'border-indigo-500 bg-indigo-500/10 hover:bg-indigo-500/20' : 'border-yellow-500 bg-yellow-500/20 hover:bg-yellow-500/30'}`}
            style={{ 
              left: `${ann.rect!.x * 100}%`, 
              top: `${ann.rect!.y * 100}%`, 
              width: `${ann.rect!.w * 100}%`, 
              height: `${ann.rect!.h * 100}%` 
            }}
            title={ann.userPrompt || "Annotation"}
          />
        );
      })}
    </div>
  );
};


// --- Main Application ---

export default function InsightPDFApp() {
  // State
  const [pdfLib, setPdfLib] = useState<any>(null); 
  const [pdfModificationLib, setPdfModificationLib] = useState<any>(null); 
  const [pdfDoc, setPdfDoc] = useState<any>(null);
  const [fileBuffer, setFileBuffer] = useState<ArrayBuffer | null>(null); 
  
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [scale, setScale] = useState<number>(1.0);
  
  const [apiKey, setApiKey] = useState<string>('');
  const [provider, setProvider] = useState<LLMProvider>('gemini');
  
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [isExporting, setIsExporting] = useState(false);
  
  const [interactionMode, setInteractionMode] = useState<InteractionMode>('read'); 
  const [selectionRect, setSelectionRect] = useState<Rect | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [activePage, setActivePage] = useState<number | null>(null); 
  const startPos = useRef<{ x: number, y: number }>({ x: 0, y: 0 });
  
  // Sidebar State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [sidebarWidth, setSidebarWidth] = useState(384);
  const [isResizingSidebar, setIsResizingSidebar] = useState(false);

  // Editing & Navigation State
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editNoteText, setEditNoteText] = useState('');
  const [targetAnnotationId, setTargetAnnotationId] = useState<string | null>(null);

  // Modal State
  const [showAnnotationModal, setShowAnnotationModal] = useState(false);
  const [pendingAnnotation, setPendingAnnotation] = useState<Partial<Annotation> | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [promptInput, setPromptInput] = useState('');
  const [noteInput, setNoteInput] = useState('');

  // Refs
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const pageRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  // --- Initialization ---
  useEffect(() => {
    const loadLibs = async () => {
      if ((window as any).pdfjsLib) {
        setPdfLib((window as any).pdfjsLib);
      } else {
        const script = document.createElement('script');
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
        script.onload = () => {
          const lib = (window as any).pdfjsLib;
          lib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
          setPdfLib(lib);
        };
        document.body.appendChild(script);
      }

      if ((window as any).PDFLib) {
        setPdfModificationLib((window as any).PDFLib);
      } else {
        const scriptMod = document.createElement('script');
        scriptMod.src = "https://unpkg.com/pdf-lib@1.17.1/dist/pdf-lib.min.js";
        scriptMod.onload = () => {
          setPdfModificationLib((window as any).PDFLib);
        };
        document.body.appendChild(scriptMod);
      }
      
      const style = document.createElement('style');
      style.innerHTML = `
        .textLayer { position: absolute; left: 0; top: 0; right: 0; bottom: 0; overflow: hidden; opacity: 0.2; line-height: 1.0; transform-origin: 0% 0%; z-index: 10; }
        .textLayer ::selection { background: rgba(0, 0, 255, 1); color: transparent; }
        .textLayer > span { color: transparent; position: absolute; white-space: pre; cursor: text; transform-origin: 0% 0%; }
      `;
      document.head.appendChild(style);
    };
    loadLibs();
  }, []);

  const onFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !pdfLib) return;
    const fileReader = new FileReader();
    fileReader.onload = async function() {
      const originalBuffer = this.result as ArrayBuffer;
      const bufferForExport = originalBuffer.slice(0);
      setFileBuffer(bufferForExport);
      const typedarray = new Uint8Array(originalBuffer);
      try {
        const loadingTask = pdfLib.getDocument(typedarray);
        const doc = await loadingTask.promise;
        setPdfDoc(doc);
        setNumPages(doc.numPages);
        setCurrentPage(1);
        setAnnotations([]); 
      } catch (error) {
        console.error(error);
        alert("Error loading PDF.");
      }
    };
    fileReader.readAsArrayBuffer(file);
  };

  // --- Scroll & Navigation Logic ---
  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const containerTop = scrollContainerRef.current.scrollTop;
    const containerHeight = scrollContainerRef.current.clientHeight;
    const centerLine = containerTop + (containerHeight / 2);

    let closestPage = 1;
    let minDistance = Infinity;

    pageRefs.current.forEach((el, pageNum) => {
      const elCenter = el.offsetTop + (el.offsetHeight / 2);
      const distance = Math.abs(centerLine - elCenter);
      if (distance < minDistance) {
        minDistance = distance;
        closestPage = pageNum;
      }
    });

    if (closestPage !== currentPage) {
      setCurrentPage(closestPage);
    }
  };

  const scrollToPage = (pageNum: number) => {
    const el = pageRefs.current.get(pageNum);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setCurrentPage(pageNum);
    }
  };

  const jumpToAnnotation = (ann: Annotation) => {
    const el = pageRefs.current.get(ann.pageNumber);
    if (el) {
      const pageHeight = el.offsetHeight;
      const topOffset = el.offsetTop + (ann.rect.y * pageHeight);
      scrollContainerRef.current?.scrollTo({
        top: Math.max(0, topOffset - 100),
        behavior: 'smooth'
      });
    }
  };

  // --- Sidebar Resize Logic ---
  const startResizing = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizingSidebar(true);
  };

  const stopResizing = () => {
    setIsResizingSidebar(false);
  };

  const resizeSidebar = useCallback((e: MouseEvent) => {
    if (isResizingSidebar) {
      const newWidth = document.body.clientWidth - e.clientX;
      setSidebarWidth(Math.max(250, Math.min(800, newWidth))); 
    }
  }, [isResizingSidebar]);

  useEffect(() => {
    window.addEventListener('mousemove', resizeSidebar);
    window.addEventListener('mouseup', stopResizing);
    return () => {
      window.removeEventListener('mousemove', resizeSidebar);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [resizeSidebar]);

  // --- Interactions ---
  const handlePageTextSelection = (e: React.MouseEvent, pageNum: number, pageEl: HTMLDivElement) => {
    if (interactionMode === 'read' || interactionMode === 'crop') return;
    
    const selection = window.getSelection();
    if (!selection || selection.toString().trim().length === 0) return;

    const range = selection.getRangeAt(0);
    const pageRect = pageEl.getBoundingClientRect();
    const boundingRect = range.getBoundingClientRect();

    const normalizedBoundingRect = {
      x: (boundingRect.left - pageRect.left) / pageRect.width,
      y: (boundingRect.top - pageRect.top) / pageRect.height,
      w: boundingRect.width / pageRect.width,
      h: boundingRect.height / pageRect.height
    };

    const clientRects = Array.from(range.getClientRects());
    const normalizedRects = clientRects.map(r => ({
      x: (r.left - pageRect.left) / pageRect.width,
      y: (r.top - pageRect.top) / pageRect.height,
      w: r.width / pageRect.width,
      h: r.height / pageRect.height
    }));

    setPendingAnnotation({ 
      type: 'text_highlight', 
      pageNumber: pageNum, 
      selectedText: selection.toString(),
      rect: normalizedBoundingRect,
      rects: normalizedRects 
    });
    setShowAnnotationModal(true);
  };

  const handlePageMouseDown = (e: React.MouseEvent, pageNum: number) => {
    if (interactionMode !== 'crop') return;
    const el = pageRefs.current.get(pageNum);
    if (!el) return;
    const rect = el.getBoundingClientRect();
    startPos.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    setIsDragging(true);
    setActivePage(pageNum);
    setSelectionRect({ x: startPos.current.x, y: startPos.current.y, w: 0, h: 0 });
  };

  const handlePageMouseMove = (e: React.MouseEvent, pageNum: number) => {
    if (!isDragging || activePage !== pageNum) return;
    const el = pageRefs.current.get(pageNum);
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    setSelectionRect({
      x: Math.min(startPos.current.x, currentX),
      y: Math.min(startPos.current.y, currentY),
      w: Math.abs(currentX - startPos.current.x),
      h: Math.abs(currentY - startPos.current.y)
    });
  };

  const handlePageMouseUp = async (e: React.MouseEvent, pageNum: number) => {
    if (!isDragging || !selectionRect || activePage !== pageNum) return;
    setIsDragging(false);
    setActivePage(null);
    if (selectionRect.w < 10 || selectionRect.h < 10) { setSelectionRect(null); return; }
    
    const el = pageRefs.current.get(pageNum);
    const canvas = el?.querySelector('canvas');
    if (canvas) {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = selectionRect.w;
      tempCanvas.height = selectionRect.h;
      const ctx = tempCanvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(canvas, selectionRect.x, selectionRect.y, selectionRect.w, selectionRect.h, 0, 0, selectionRect.w, selectionRect.h);
        setPendingAnnotation({
          type: 'region_crop',
          pageNumber: pageNum,
          rect: { x: selectionRect.x / canvas.width, y: selectionRect.y / canvas.height, w: selectionRect.w / canvas.width, h: selectionRect.h / canvas.height },
          imageBase64: tempCanvas.toDataURL('image/png')
        });
        setShowAnnotationModal(true);
      }
    }
    setSelectionRect(null);
  };

  // --- Annotation & Logic ---
  const handleLLMSubmit = async (actionType: 'explain' | 'summarize' | 'custom') => {
    if (!apiKey) { alert("Please enter an API Key."); return; }
    setIsProcessing(true);
    try {
      const systemPrompt = "You are a helpful academic assistant explaining PDF content.";
      let query = promptInput;
      if (actionType === 'explain') query = "Explain this in detail, focusing on clarity and key concepts.";
      else if (actionType === 'summarize') query = "Provide a concise summary of this.";
      else if (actionType === 'custom' && !query.trim()) query = "Explain this.";

      const context = { text: pendingAnnotation?.selectedText, image: pendingAnnotation?.imageBase64 };
      
      let responseText = "";
      if (provider === 'gemini') responseText = await callGemini(apiKey, systemPrompt, query, context);
      else if (provider === 'openai') responseText = await callOpenAI(apiKey, systemPrompt, query, context);
      else if (provider === 'anthropic') responseText = await callAnthropic(apiKey, systemPrompt, query, context);

      setAnnotations(prev => [...prev, {
        id: Date.now().toString(),
        pageNumber: pendingAnnotation!.pageNumber!,
        type: pendingAnnotation!.type!,
        selectedText: pendingAnnotation?.selectedText,
        rect: pendingAnnotation!.rect!, 
        rects: pendingAnnotation?.rects,
        imageBase64: pendingAnnotation?.imageBase64,
        userPrompt: actionType === 'custom' ? promptInput : actionType,
        llmResponse: responseText,
        manualNote: noteInput,
        timestamp: Date.now(),
        providerUsed: provider
      }]);
      closeModal();
    } catch (e: any) {
      alert(`Error: ${e.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleManualSave = () => {
    setAnnotations(prev => [...prev, {
      id: Date.now().toString(),
      pageNumber: pendingAnnotation!.pageNumber!,
      type: pendingAnnotation!.type!,
      selectedText: pendingAnnotation?.selectedText,
      rect: pendingAnnotation!.rect!,
      rects: pendingAnnotation?.rects,
      imageBase64: pendingAnnotation?.imageBase64,
      userPrompt: "Manual Note Only",
      llmResponse: "",
      manualNote: noteInput || "No content",
      timestamp: Date.now(),
      providerUsed: undefined
    }]);
    closeModal();
  };

  // --- PDF Export ---
  const handleDownloadPdf = async () => {
    if (!fileBuffer || !pdfModificationLib) {
      if (!pdfModificationLib) alert("PDF Mod Library not loaded yet");
      return;
    }
    setIsExporting(true);
    try {
      const { PDFDocument, rgb, StandardFonts } = pdfModificationLib;
      const pdfDoc = await PDFDocument.load(fileBuffer);
      const pages = pdfDoc.getPages();
      const helveticaBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);
      const helvetica = await pdfDoc.embedFont(StandardFonts.Helvetica);

      annotations.forEach((ann: Annotation) => {
        if (ann.pageNumber <= pages.length) {
          const page = pages[ann.pageNumber - 1];
          const { width, height } = page.getSize();
          const rectsToDraw = (ann.rects && ann.rects.length > 0) ? ann.rects : [ann.rect];
          rectsToDraw.forEach((r: Rect) => {
            page.drawRectangle({
              x: r.x * width, y: height - (r.y * height) - (r.h * height), 
              width: r.w * width, height: r.h * height,
              color: rgb(1, 1, 0), opacity: 0.35,
            });
          });
          page.drawText(`#${annotations.indexOf(ann) + 1}`, {
            x: (rectsToDraw[0].x * width) - 15, y: height - (rectsToDraw[0].y * height) - 10,
            size: 10, font: helveticaBold, color: rgb(1, 0, 0),
          });
        }
      });

      let summaryPage = pdfDoc.addPage();
      let { width, height } = summaryPage.getSize();
      let yOffset = height - 50;
      summaryPage.drawText('InsightPDF Summary', { x: 50, y: yOffset, size: 24, font: helveticaBold, color: rgb(0, 0, 0) });
      yOffset -= 40;

      for (let i = 0; i < annotations.length; i++) {
        const ann = annotations[i];
        if (yOffset < 100) { summaryPage = pdfDoc.addPage(); yOffset = height - 50; }
        
        summaryPage.drawText(`#${i + 1} [Page ${ann.pageNumber}]`, { x: 50, y: yOffset, size: 12, font: helveticaBold });
        yOffset -= 15;
        if (ann.userPrompt) {
           summaryPage.drawText(`Q: ${ann.userPrompt}`, { x: 50, y: yOffset, size: 10, font: helveticaBold });
           yOffset -= 12;
        }
        if (ann.llmResponse) {
           const words = ann.llmResponse.replace(/\n/g, ' ').split(' ');
           let line = 'AI: ';
           for (const word of words) {
             if (line.length + word.length > 90) {
                summaryPage.drawText(line, { x: 50, y: yOffset, size: 10, font: helvetica });
                yOffset -= 12;
                line = word + ' ';
             } else { line += word + ' '; }
           }
           summaryPage.drawText(line, { x: 50, y: yOffset, size: 10, font: helvetica });
           yOffset -= 12;
        }
        yOffset -= 20;
      }

      const pdfBytes = await pdfDoc.save();
      downloadBlob(new Blob([pdfBytes], { type: 'application/pdf' }), 'annotated_document.pdf');
    } catch (err) {
      console.error("Export failed", err);
      alert("Failed to generate PDF.");
    } finally {
      setIsExporting(false);
    }
  };

  const closeModal = () => {
    setShowAnnotationModal(false);
    setPendingAnnotation(null);
    setPromptInput('');
    setNoteInput('');
  };

  const startEditing = (ann: Annotation) => { setEditingId(ann.id); setEditNoteText(ann.manualNote); };
  const saveEdit = (id: string) => { setAnnotations(prev => prev.map(a => a.id === id ? { ...a, manualNote: editNoteText } : a)); setEditingId(null); };

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-slate-800 font-sans overflow-hidden">
      <header className="bg-white border-b border-gray-200 h-16 flex items-center justify-between px-4 shadow-sm z-30 relative shrink-0">
        <div className="flex items-center space-x-4">
          <div className="flex items-center text-indigo-600 font-bold text-xl">
            <BrainCircuit className="mr-2" /> InsightPDF
          </div>
          <label className="flex items-center space-x-2 px-3 py-1.5 bg-indigo-50 text-indigo-700 rounded-md cursor-pointer hover:bg-indigo-100 transition">
            <Upload size={16} />
            <span className="text-sm font-medium">Upload</span>
            <input type="file" accept=".pdf" onChange={onFileChange} className="hidden" />
          </label>
          {annotations.length > 0 && (
            <button 
              onClick={handleDownloadPdf}
              disabled={isExporting || !pdfModificationLib}
              className="flex items-center space-x-2 px-3 py-1.5 bg-white border border-gray-300 text-gray-700 rounded-md cursor-pointer hover:bg-gray-50 transition disabled:opacity-50 shrink-0"
            >
              {isExporting ? <Loader2 className="animate-spin" size={16}/> : <Download size={16} />}
              <span className="text-sm font-medium">Export PDF</span>
            </button>
          )}
        </div>

        <div className="flex-1 max-w-2xl mx-4 flex space-x-2 min-w-0">
          <div className="relative shrink-0">
            <select 
              value={provider} 
              onChange={(e) => setProvider(e.target.value as LLMProvider)}
              className="h-full pl-9 pr-8 py-1.5 text-sm border border-gray-300 rounded-md bg-gray-50 focus:ring-2 focus:ring-indigo-500 focus:outline-none appearance-none cursor-pointer hover:bg-gray-100 font-medium"
            >
              <option value="gemini">Google Gemini</option>
              <option value="openai">OpenAI (GPT-4)</option>
              <option value="anthropic">Anthropic (Claude 3.5)</option>
            </select>
            <Settings size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"/>
          </div>
          <input 
            type="password"
            placeholder={`Enter ${provider === 'gemini' ? 'Gemini' : provider === 'openai' ? 'OpenAI' : 'Anthropic'} API Key...`} 
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="flex-1 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:outline-none min-w-[100px]"
          />
        </div>

        <div className="flex items-center space-x-2 shrink-0">
          <div className="bg-gray-100 rounded-lg p-1 flex space-x-1">
            <button onClick={() => setInteractionMode('read')} className={`p-2 rounded-md transition ${interactionMode === 'read' ? 'bg-white shadow-sm text-indigo-600 ring-1 ring-indigo-200' : 'text-gray-500 hover:bg-gray-200'}`} title="Read Mode"><MousePointer2 size={18} /></button>
            <button onClick={() => setInteractionMode('ask')} className={`p-2 rounded-md transition ${interactionMode === 'ask' ? 'bg-white shadow-sm text-indigo-600 ring-1 ring-indigo-200' : 'text-gray-500 hover:bg-gray-200'}`} title="Ask AI Mode"><Sparkles size={18} /></button>
            <button onClick={() => setInteractionMode('crop')} className={`p-2 rounded-md transition ${interactionMode === 'crop' ? 'bg-white shadow-sm text-indigo-600 ring-1 ring-indigo-200' : 'text-gray-500 hover:bg-gray-200'}`} title="Crop Mode"><Crop size={18} /></button>
          </div>
          <div className="h-6 w-px bg-gray-300 mx-2"></div>
          <button onClick={() => setScale(s => Math.max(0.5, s - 0.2))} className="p-2 hover:bg-gray-100 rounded-full"><ZoomOut size={18}/></button>
          <span className="text-sm w-12 text-center">{Math.round(scale * 100)}%</span>
          <button onClick={() => setScale(s => Math.min(3, s + 0.2))} className="p-2 hover:bg-gray-100 rounded-full"><ZoomIn size={18}/></button>
          <div className="h-6 w-px bg-gray-300 mx-2"></div>
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className={`p-2 rounded-md transition ${isSidebarOpen ? 'bg-indigo-50 text-indigo-600' : 'text-gray-500 hover:bg-gray-100'}`} title="Toggle Sidebar"><PanelRight size={18} /></button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden relative">
        <div 
          ref={scrollContainerRef}
          onScroll={handleScroll}
          className="flex-1 bg-gray-100 overflow-auto flex flex-col items-center relative p-8"
        >
          {!pdfDoc ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <Upload size={64} className="mb-4 opacity-20" />
              <p className="text-lg">Upload a PDF to get started</p>
              {!pdfLib && <p className="text-xs mt-2 text-indigo-400 animate-pulse">Initializing PDF Engine...</p>}
            </div>
          ) : (
             Array.from(new Array(numPages), (el, index) => (
               <PDFPage 
                 key={index + 1}
                 pageNum={index + 1}
                 pdfDoc={pdfDoc}
                 pdfLib={pdfLib}
                 scale={scale}
                 annotations={annotations.filter(a => a.pageNumber === index + 1)}
                 interactionMode={interactionMode}
                 onMouseDown={handlePageMouseDown}
                 onMouseMove={handlePageMouseMove}
                 onMouseUp={handlePageMouseUp}
                 onTextSelection={handlePageTextSelection}
                 selectionRect={selectionRect}
                 isInteractingWithThisPage={activePage === (index + 1)}
                 setPageRef={(p: number, ref: HTMLDivElement) => pageRefs.current.set(p, ref)}
               />
             ))
          )}
          
          {pdfDoc && (
            <div className="sticky bottom-8 bg-white/90 backdrop-blur shadow-lg rounded-full px-4 py-2 flex items-center space-x-4 z-40 border border-gray-200 mt-auto mb-4">
              <button disabled={currentPage <= 1} onClick={() => scrollToPage(currentPage - 1)} className="p-1 hover:bg-gray-100 rounded-full disabled:opacity-30"><ChevronLeft size={24} /></button>
              <span className="font-medium text-sm whitespace-nowrap">Page {currentPage} of {numPages}</span>
              <button disabled={currentPage >= numPages} onClick={() => scrollToPage(currentPage + 1)} className="p-1 hover:bg-gray-100 rounded-full disabled:opacity-30"><ChevronRight size={24} /></button>
            </div>
          )}
        </div>

        {isSidebarOpen && (
          <>
            <div
              onMouseDown={startResizing}
              className={`w-4 -ml-2 flex items-center justify-center cursor-col-resize z-40 hover:bg-indigo-50 group transition-colors ${isResizingSidebar ? 'bg-indigo-100' : ''}`}
              style={{ width: '16px' }}
            >
              <GripVertical size={12} className={`text-gray-300 group-hover:text-indigo-400 ${isResizingSidebar ? 'text-indigo-500' : ''}`} />
            </div>

            <div 
              className="bg-white border-l border-gray-200 flex flex-col shadow-xl z-30 transition-all"
              style={{ width: sidebarWidth }}
            >
              <div className="p-4 border-b border-gray-100 bg-gray-50 flex justify-between items-center shrink-0">
                <h2 className="font-bold text-gray-700 flex items-center">
                  <FileText className="mr-2" size={18} /> Annotations ({annotations.length})
                </h2>
                <button onClick={() => setIsSidebarOpen(false)} className="text-gray-400 hover:text-gray-600 md:hidden"><X size={18}/></button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {annotations.length === 0 ? (
                  <div className="text-center text-gray-400 mt-10">
                    <p className="text-sm">No annotations yet.</p>
                    <p className="text-xs mt-2">Switch to "Ask AI" mode to select text.</p>
                  </div>
                ) : (
                  annotations.map(ann => (
                    <div 
                      key={ann.id} 
                      className={`bg-white border rounded-xl p-4 shadow-sm hover:shadow-md transition group cursor-pointer ${currentPage === ann.pageNumber ? 'border-indigo-200 ring-1 ring-indigo-100' : 'border-gray-200'}`}
                      onClick={() => jumpToAnnotation(ann)}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center space-x-2">
                          <span className="text-xs font-semibold px-2 py-1 bg-indigo-50 text-indigo-700 rounded-full">Page {ann.pageNumber}</span>
                          {ann.providerUsed ? (
                            <span className="text-[10px] px-1.5 py-0.5 bg-gray-100 text-gray-500 rounded uppercase tracking-wider">{ann.providerUsed}</span>
                          ) : (
                            <span className="text-[10px] px-1.5 py-0.5 bg-yellow-100 text-yellow-700 rounded uppercase tracking-wider">Manual</span>
                          )}
                        </div>
                        <div className="flex space-x-2 opacity-50 group-hover:opacity-100 transition">
                          <button onClick={(e) => { e.stopPropagation(); startEditing(ann); }} className="text-gray-400 hover:text-indigo-600" title="Edit Note"><Edit2 size={14} /></button>
                          <button onClick={(e) => { e.stopPropagation(); setAnnotations(prev => prev.filter(a => a.id !== ann.id)); }} className="text-gray-400 hover:text-red-500"><Trash2 size={14} /></button>
                        </div>
                      </div>
                      
                      {ann.type === 'region_crop' && ann.imageBase64 && <img src={ann.imageBase64} alt="Crop" className="w-full h-32 object-contain bg-gray-900 rounded-md mb-3 border border-gray-100" />}
                      {ann.type === 'text_highlight' && ann.selectedText && <blockquote className="text-xs text-gray-500 italic border-l-2 border-indigo-300 pl-2 mb-3 line-clamp-3 bg-gray-50 p-2 rounded">"{ann.selectedText}"</blockquote>}
                      
                      {ann.llmResponse && (
                        <div className="mb-3">
                          <div className="text-xs uppercase font-bold text-gray-400 mb-1 flex items-center"><BrainCircuit size={12} className="mr-1" /> AI Explanation</div>
                          <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">{ann.llmResponse}</p>
                        </div>
                      )}
                      
                      <div className="mt-2 pt-2 border-t border-gray-100" onClick={(e) => e.stopPropagation()}>
                        {editingId === ann.id ? (
                          <div className="space-y-2">
                            <textarea 
                              value={editNoteText}
                              onChange={(e) => setEditNoteText(e.target.value)}
                              className="w-full p-2 text-sm border border-indigo-300 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
                              rows={3}
                            />
                            <div className="flex justify-end space-x-2">
                              <button onClick={() => setEditingId(null)} className="text-xs px-2 py-1 text-gray-500 hover:bg-gray-100 rounded">Cancel</button>
                              <button onClick={() => saveEdit(ann.id)} className="text-xs px-2 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 flex items-center"><SaveIcon size={12} className="mr-1"/> Save</button>
                            </div>
                          </div>
                        ) : (
                          ann.manualNote ? (
                            <p className="text-sm text-gray-600"><span className="font-semibold">Note:</span> {ann.manualNote}</p>
                          ) : (
                             !ann.llmResponse && <p className="text-sm text-gray-400 italic">No note attached</p>
                          )
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <Modal isOpen={showAnnotationModal} onClose={closeModal} title={pendingAnnotation?.type === 'region_crop' ? 'Analyze Image Region' : 'Analyze Selected Text'}>
        <div className="space-y-6">
          <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
            {pendingAnnotation?.type === 'region_crop' ? <div className="text-center"><img src={pendingAnnotation.imageBase64} alt="Selected" className="max-h-40 mx-auto shadow-sm rounded" /></div> : <p className="text-sm text-gray-600 italic">"{pendingAnnotation?.selectedText}"</p>}
          </div>
          <div className="space-y-3">
            <label className="text-sm font-semibold text-gray-700">Ask {provider === 'gemini' ? 'Gemini' : provider === 'openai' ? 'GPT-4' : 'Claude'}</label>
            <div className="grid grid-cols-2 gap-2">
              <button onClick={() => handleLLMSubmit('explain')} disabled={isProcessing} className="flex items-center justify-center px-4 py-2 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 border border-indigo-200 transition disabled:opacity-50"><BrainCircuit size={16} className="mr-2" /> Explain</button>
              <button onClick={() => handleLLMSubmit('summarize')} disabled={isProcessing} className="flex items-center justify-center px-4 py-2 bg-indigo-50 text-indigo-700 rounded-lg hover:bg-indigo-100 border border-indigo-200 transition disabled:opacity-50"><FileText size={16} className="mr-2" /> Summarize</button>
            </div>
            <div className="relative">
              <input type="text" value={promptInput} onChange={(e) => setPromptInput(e.target.value)} placeholder="Or ask a custom question..." className="w-full pl-4 pr-12 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" />
              <button onClick={() => handleLLMSubmit('custom')} disabled={isProcessing || !promptInput} className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-indigo-600 hover:bg-indigo-50 rounded disabled:opacity-50">{isProcessing ? <Loader2 className="animate-spin" size={18}/> : <MessageSquarePlus size={18} />}</button>
            </div>
          </div>
          <div className="border-t border-gray-200 pt-4">
            <label className="text-sm font-semibold text-gray-700 mb-2 block">Manual Note (Optional)</label>
            <textarea value={noteInput} onChange={(e) => setNoteInput(e.target.value)} placeholder="Add your own thoughts here..." className="w-full p-3 border border-gray-300 rounded-lg h-24 focus:ring-2 focus:ring-indigo-500 focus:outline-none resize-none" />
          </div>
          <div className="flex justify-end space-x-3 pt-2">
            <button onClick={handleManualSave} className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg transition">Save Note Only</button>
          </div>
        </div>
      </Modal>
    </div>
  );
}