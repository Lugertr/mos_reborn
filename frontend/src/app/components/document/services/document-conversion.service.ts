// document-conversion.service.ts
import { Injectable } from '@angular/core';

export type DocumentValue = {
  name: string;
  type: UserDocumentType;
  size: number;
  base64: string;
};

export enum UserDocumentType {
  Image,
  Pdf,
}

/**
 * сервис конвертации файлов в base64
 *
 * - Изображения -> data:image/...
 * - PDF -> data:application/pdf;base64,... (файл в base64, НЕ картинка)
 */
@Injectable({ providedIn: 'root' })
export class DocumentConversionService {
  private readonly supportedImagePattern = /^image\//i;

  /**
   * Конвертирует файл в dataURL.
   * Для PDF вернёт data:application/pdf;base64,... (не картинку).
   * Бросает Error при проблеме.
   */
  async convertFileToDataUrl(file?: File): Promise<DocumentValue> {
    if (!file) {
      throw new Error('Файл не передан');
    }

    // Проверка типа: изображения или pdf — остальные типы считаются неподдерживаемыми.
    if (this.isImage(file) || this.isPdf(file)) {
      try {
        const base64 = await this.fileToDataUrl(file);
        return {
          name: file.name,
          type: this.isPdf(file) ? UserDocumentType.Pdf : UserDocumentType.Image,
          size: file.size,
          base64
        };
      } catch (e) {
        throw new Error(`Ошибка чтения файла: ${String(e ?? 'unknown')}`);
      }
    }

    throw new Error(`Неподдерживаемый тип файла: ${file.type || file.name}`);
  }

  isImage(file: File): boolean {
    return this.supportedImagePattern.test(file.type) || /\.(jpe?g|png|gif|webp|svg|heic)$/i.test(file.name);
  }

  isPdf(file: File): boolean {
    return file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
  }

  private fileToDataUrl(file: File): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        const res = reader.result;
        if (typeof res === 'string') {
          resolve(res);
        } else {
          reject(new Error('FileReader вернул не строку'));
        }
      };

      reader.onerror = (ev) => {
        reader.abort();
        reject(new Error('Ошибка при чтении файла'));
      };

      reader.readAsDataURL(file);
    });
  }
}

export function dataURLToBlob(dataUrl: string): Blob {
  const [meta, b64] = dataUrl.split(',');
  const contentTypeMatch = meta.match(/data:([^;]+);base64/);
  const contentType = contentTypeMatch ? contentTypeMatch[1] : '';
  const binary = atob(b64);
  const len = binary.length;
  const buffer = new Uint8Array(len);
  for (let i = 0; i < len; i++) buffer[i] = binary.charCodeAt(i);
  return new Blob([buffer], { type: contentType });
}
