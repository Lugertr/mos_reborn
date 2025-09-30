import { Component, inject, input, signal, Output, EventEmitter, ChangeDetectionStrategy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DragDropModule } from '@angular/cdk/drag-drop';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { InformerService } from '@core/services/informer.service';

type Rejection = { file?: File; reason?: string, ok?: boolean };

@Component({
  selector: 'app-file-drop',
  standalone: true,
  imports: [CommonModule, DragDropModule, MatProgressBarModule, MatButtonModule, MatIconModule],
  templateUrl: './file-drop.component.html',
  styleUrls: ['./file-drop.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class FileDropComponent {
  private readonly informerSrv = inject(InformerService);

  maxFiles = input<number>(1);
  /**
   * allowedTypes: массив MIME-типов, например ['image/png','image/jpeg'].
   * Если null/undefined — проверка по типу не выполняется.
   */
  allowedTypes = input<string[] | null>(null);
  maxFileSize = input<number | null>(null);
  /**
   * allowedExtensions: массив расширений без точки, нижнего регистра, например ['png','jpg'].
   * Если null — проверка по расширению не выполняется.
   */
  allowedExtensions = input<string[] | null>(null);

  files = signal<File[]>([]);
  uploadProgress = signal<{ [name: string]: number }>({});

  @Output() filesChange = new EventEmitter<File[]>();

  onDrop(event: DragEvent): void {
    event.preventDefault();
    const dropped = Array.from(event.dataTransfer?.files ?? []);
    this.processIncomingFiles(dropped);
  }

  onFileInput(event: Event): void {
    const input = event.target as HTMLInputElement;
    const selected = Array.from(input.files ?? []);
    this.processIncomingFiles(selected);
    input.value = '';
  }

  private processIncomingFiles(newFiles: File[]): void {
    if (!newFiles.length) return;

    const current = this.files();
    const spaceLeft = Math.max(0, this.maxFiles() - current.length);
    if (spaceLeft <= 0) {
      this.informerSrv.warn(`Добавлено максимальное количество файлов (${this.maxFiles()})`);
      return;
    }

    const rejections: Rejection[] = [];
    const accepted: File[] = [];

    for (const f of newFiles) {
      const res = this.validateFile(f);
      if (!res.ok) {
        rejections.push({ file: f, reason: res.reason! });
      } else {
        accepted.push(f);
      }
    }

    let willAdd = accepted;
    if (accepted.length > spaceLeft) {
      willAdd = accepted.slice(0, spaceLeft);
      const droppedCount = accepted.length - willAdd.length;
      rejections.push({
        file: accepted[accepted.length - 1],
        reason: `Лимит файлов — можно добавить только ещё ${droppedCount}`,
      });
    }

    if (rejections.length) {
      const msgs = rejections
        .slice(0, 6)
        .map(r => `${r.file.name}: ${r.reason}`)
        .join('\n');
      const more = rejections.length > 6 ? `\n...и ещё ${rejections.length - 6} файлов отклонено` : '';
      this.informerSrv.warn(`Некоторые файлы не прошли валидацию:\n${msgs}${more}`);
    }

    if (willAdd.length) {
      this.files.set([...current, ...willAdd]);
      willAdd.forEach(f => this.startUpload(f));
      this.emitFilesChange();
    }
  }

  private validateFile(file: File): Rejection {
    const types = this.allowedTypes();
    if (types && types.length) {
      if (!file.type || !types.includes(file.type)) {
        const ext = this.fileExt(file);
        const extMatchesMime = types.some(t => t.split('/').pop() === ext);
        if (!extMatchesMime) {
          return { ok: false, reason: `Недопустимый тип (ожидалось: ${types.join(', ')})` };
        }
      }
    }

    const allowedExt = this.allowedExtensions();
    if (allowedExt && allowedExt.length) {
      const ext = this.fileExt(file);
      if (!ext || !allowedExt.includes(ext)) {
        return { ok: false, reason: `Недопустимое расширение (ожидалось: ${allowedExt.join(', ')})` };
      }
    }

    const maxSize = this.maxFileSize();
    if (maxSize != null && file.size > maxSize) {
      return { ok: false, reason: `Файл слишком большой (максимум ${this.humanSize(maxSize)})` };
    }

    return { ok: true };
  }

  removeFile(file: File): void {
    const updated = this.files().filter(f => !(f.name === file.name && f.size === file.size && f.lastModified === file.lastModified));
    this.files.set(updated);
    this.uploadProgress.update(mp => {
      const { [file.name]: _, ...rest } = mp;
      return { ...rest };
    });
    this.emitFilesChange();
  }

  private emitFilesChange(): void {
    this.filesChange.emit([...this.files()]);
  }

  private startUpload(file: File): void {
    const name = file.name;
    this.uploadProgress.update(mp => {
      mp[name] = 0;
      return { ...mp };
    });
    const interval = setInterval(() => {
      this.uploadProgress.update(mp => {
        const val = Math.min((mp[name] ?? 0) + 10, 100);
        return { ...mp, [name]: val };
      });
      if (this.uploadProgress()[name] >= 100) {
        clearInterval(interval);
      }
    }, 300);
  }

  private fileExt(file: File): string {
    const name = file.name || '';
    const idx = name.lastIndexOf('.');
    if (idx === -1) return '';
    return name.slice(idx + 1).toLowerCase();
  }

  private humanSize(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    let val = bytes;
    while (val >= 1024 && i < units.length - 1) {
      val /= 1024;
      i++;
    }
    return `${Math.round(val * 10) / 10} ${units[i]}`;
  }
}
