// progress.ts
export interface ProgressStatus {
  text: string;
  percent: number; // 0..100
}

// фазы ровно как шлёт бэкенд
export enum OcrPhase {
  Start             = 'start',
  Received          = 'received',
  OsdRotation       = 'osd_rotation',
  Preprocess        = 'preprocess',
  SegmentLines      = 'segment_lines',
  SegmentLinesDone  = 'segment_lines_done',
  TrocrFallback     = 'trocr_fallback',
  Assembling        = 'assembling',
  Done              = 'done',
  Error             = 'error',
}

// дефолтные проценты для фаз (с твоих примеров)
export const STATUS_MAP: Map<OcrPhase, ProgressStatus> = new Map([
  [OcrPhase.Start,            { text: 'Запуск…',               percent: 1   }],
  [OcrPhase.Received,         { text: 'Файл получен',          percent: 3   }],
  [OcrPhase.OsdRotation,      { text: 'Определение поворота…', percent: 8   }],
  [OcrPhase.Preprocess,       { text: 'Предобработка…',        percent: 18  }],
  [OcrPhase.SegmentLines,     { text: 'Сегментация строк…',    percent: 35  }],
  [OcrPhase.SegmentLinesDone, { text: 'Сегментация завершена', percent: 55  }],
  [OcrPhase.TrocrFallback,    { text: 'TrOCR для сложных…',    percent: 65  }],
  [OcrPhase.Assembling,       { text: 'Сборка результата…',    percent: 88  }],
  [OcrPhase.Done,             { text: 'Готово',                percent: 100 }],
  [OcrPhase.Error,            { text: 'Ошибка',                percent: 100 }],
]);

// входящий прогресс теперь понимает обе схемы: phase/progress/note и step/percent/message
export type OcrProgress = {
  phase?: string;     // <= главная «фаза» от бэка
  progress?: number;  // <= 0..100 от бэка
  note?: string;      // <= текст от бэка
  // поддерживаем старые/альтернативные ключи:
  step?: string;
  percent?: number;
  message?: string;
  current?: number;
  total?: number;
  [k: string]: any;
};

export function resolveProgress(p?: OcrProgress): ProgressStatus {
  if (!p) return { text: 'Обработка…', percent: 0 };

  // 1) приоритет — progress (от бэка)
  if (isNum(p.progress)) {
    return {
      text:  mapTextByPhaseOrStep(p.phase || p.step) || 'Обработка…',
      percent: clamp(Math.round(p.progress!)),
    };
  }
  // 2) percent (альтернативный ключ)
  if (isNum(p.percent)) {
    return {
      text:  mapTextByPhaseOrStep(p.phase || p.step) || 'Обработка…',
      percent: clamp(Math.round(p.percent!)),
    };
  }
  // 3) current/total (если прислали так)
  if (isNum(p.current) && isNum(p.total) && p.total! > 0) {
    const pc = clamp(Math.round((p.current! / p.total!) * 100));
    return {
      text:  mapTextByPhaseOrStep(p.phase || p.step) || 'Обработка…',
      percent: pc,
    };
  }
  // 4) нет числа — берём дефолт из карты по фазе/степу
  const byStep = mapByPhaseOrStep(p.phase || p.step);
  if (byStep) return byStep;

  return { text: 'Обработка…', percent: 0 };
}

// --- helpers ---
function toPhase(s?: string): OcrPhase | undefined {
  if (!s) return null;
  const key = s.toLowerCase().trim() as OcrPhase;
  return (Object.values(OcrPhase) as string[]).includes(key) ? (key as OcrPhase) : undefined;
}
function mapByPhaseOrStep(s?: string): ProgressStatus | undefined {
  const ph = toPhase(s);
  return ph ? STATUS_MAP.get(ph) : undefined;
}
function mapTextByPhaseOrStep(s?: string): string | undefined {
  return mapByPhaseOrStep(s)?.text;
}
function isNum(n: any): n is number {
  return typeof n === 'number' && Number.isFinite(n);
}
function clamp(n: number) {
  return Math.max(0, Math.min(100, n));
}
