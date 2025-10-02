/**
 * валидация и подготовка файлов изображений для OCR:
 *  - проверка формата/размера
 *  - вычисление натуральных размеров [width, height] (важно для корректной отправки в бекенд)
 */
import { inject, Injectable } from '@angular/core';
import { isFilledArray } from '@core/helpers/array-utils';
import { InformerService } from '@core/services/informer.service';

const MAX_IMAGE_SIZE = 10 * 1024 * 1024;

export interface FileData {
  name: string;
  file: File;
  size: [number, number]; // [width, height]
}

@Injectable()
export class FileConversionService {
  private readonly informerSrv = inject(InformerService);

  private readonly supportedImagePattern = /^image\//i;

  isImage(file: File): boolean {
    return (
      this.supportedImagePattern.test(file.type) ||
      /\.(jpe?g|png|gif|webp|svg|heic)$/i.test(file.name)
    );
  }

  async onSelectFiles(ev: Event): Promise<FileData[]> {
    const input = ev.target as HTMLInputElement | null;
    const files = input?.files;
    if (!files?.length) return [];
    const readers: Promise<FileData | null>[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files.item(i);
      readers.push(this.handleFileUpload(file));
    }

    const list = await Promise.all(readers);

    const correctImgData = list.filter(Boolean) as FileData[];
    if (isFilledArray(correctImgData)) {
      this.informerSrv.info(`Успешно загрузилось ${correctImgData.length} файлов`);
    } else {
      this.informerSrv.warn('Не удалось загрузить ни одного файла, попробуйте повторить попытку');
    }
    if (input) input.value = '';

    return correctImgData;
  }

  private async handleFileUpload(file: File | null): Promise<FileData | null> {
    try {
      if (!file) {
        throw new Error(`Файл не обнаружен`);
      }
      if (!this.isImage(file)) {
        throw new Error('Некорректный формат файла');
      }
      const tooLarge = file.size > MAX_IMAGE_SIZE;
      if (tooLarge) {
        throw new Error(`Файл ${file.name} слишком тяжёлый, максимальный размер файла: 10 МБ`);
      }
      const size = await this.getImageSize(file); // [width, height]
      return { name: file.name, file, size };
    } catch (err) {
      this.informerSrv.error(err);
      return null;
    }
  }

  private async getImageSize(file: File): Promise<[number, number]> {
    if (!file.type.startsWith('image/')) {
      throw new Error('Файл не является изображением');
    }

    try {
      if (typeof createImageBitmap === 'function') {
        const bitmap = await createImageBitmap(file);
        const size: [number, number] = [bitmap.width, bitmap.height];
        try { bitmap.close?.(); } catch { /* noop */ }
        return size;
      }
    } catch { }

    return await new Promise<[number, number]>((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        URL.revokeObjectURL(url);
        resolve([w, h]);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Не удалось определить размеры изображения'));
      };
      img.src = url;
    });
  }
}
