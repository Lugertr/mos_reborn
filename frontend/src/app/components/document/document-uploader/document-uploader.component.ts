import { Component, ChangeDetectionStrategy, EventEmitter, Output, output, inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { isFilledArray } from '@core/helpers/array-utils';
import { InformerService } from '@core/services/informer.service';

const MAX_IMAGE_SIZE = 10 * 1024 * 1024;

export interface FileData { name: string; dataUrl: string }

@Component({
  selector: 'app-document-uploader',
  standalone: true,
  imports: [MatButtonModule, MatIconModule],
  templateUrl: './document-uploader.component.html',
  styleUrls: ['./document-uploader.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DocumentUploaderComponent {
  readonly filesSelected = output<FileData[]>();
  private readonly informerSrv = inject(InformerService);

  private readonly supportedImagePattern = /^image\//i;

  isImage(file: File): boolean {
    return this.supportedImagePattern.test(file.type) || /\.(jpe?g|png|gif|webp|svg|heic)$/i.test(file.name);
  }

  async onSelectFiles(ev: Event): Promise<void> {
    const input = ev.target as HTMLInputElement | null;
    const files = input?.files;
    if (!files?.length) return;
    const readers: Promise<FileData>[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files.item(i);
      readers.push(this.handleFileUpload(file));

    }

    Promise.all(readers).then(list => {
      const correctImgData = list.filter(Boolean);
      if (isFilledArray(correctImgData)) {
        this.informerSrv.info(`Успешно загрузилось ${correctImgData.length} файлов`);
        this.filesSelected.emit(list);
      } else {
        this.informerSrv.warn('Не удалось загрузить ни одного файла, попробуйте повторить попытку');
      }
    });
    input.value = '';
  }

  async onFile(event: Event): Promise<void> {
    const inputEl = event.target as HTMLInputElement | null;
    if (!inputEl || !inputEl.files || inputEl.files.length === 0) { return; }
    const f = inputEl.files[0];
    inputEl.value = '';
    await this.handleFileUpload(f);
  }

  private async handleFileUpload(file: File | null): Promise<FileData> {
    try {
      if (!file) {
        throw new Error(`Файл не обнаружен`);
      }

      if (!this.isImage(file)) {
        throw new Error('Некорректный формат файла');
      }

      const tooLarge = file.size > MAX_IMAGE_SIZE;
      if (tooLarge) {
        throw new Error(`Файл ${file.name} слишком тяжелый, максимальный размер файла: 10 МБ`);
      }

      const base64 = await this.fileToDataUrl(file);

      if (!base64) {
        throw new Error(`Не удалось обработать файл ${file.name}`);
      }
      return Promise.resolve({
        name: file.name,
        dataUrl: base64
      });
    } catch (err) {
      this.informerSrv.error(err);
      return Promise.resolve(null);
    }
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
