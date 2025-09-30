import {
  Component,
  ChangeDetectionStrategy,
  forwardRef,
  inject,

  input,
  output,
  signal,
  HostListener,
  ElementRef,
  viewChild,
} from '@angular/core';
import {
  AbstractControl,
  ControlValueAccessor,
  NG_VALUE_ACCESSOR,
  ValidationErrors,
} from '@angular/forms';
import { MatDialogModule } from '@angular/material/dialog';
import { DocumentPreviewComponent } from '../document-preview/document-preview.component';
import { DocumentConversionService, DocumentValue, UserDocumentType } from '../services/document-conversion.service';
import { InformerService } from '@core/services/informer.service';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-document-reader',
  standalone: true,
  imports: [MatDialogModule, MatButtonModule, MatIconModule, DocumentPreviewComponent],
  providers: [
    {
      provide: NG_VALUE_ACCESSOR,
      useExisting: forwardRef(() => DocumentReaderComponent),
      multi: true,
    },
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './document-reader.component.html',
  styleUrl: './document-reader.component.scss',
  host: { '[class.pointer]': '!this.value()' }
})
export class DocumentReaderComponent implements ControlValueAccessor {
  fileInput = viewChild<ElementRef<HTMLInputElement>>('fileUploader');
  readonly maxImageSize = input<number>(5 * 1024 * 1024);
  readonly maxPdfSize = input<number>(10 * 1024 * 1024);

  readonly valueChange = output<DocumentValue | null>();

  readonly value = signal<DocumentValue | null>(null);
  readonly errors = signal<ValidationErrors | null>(null);
  readonly disabled = signal<boolean>(false);

  private onChange: (v: DocumentValue | null) => void = (): void => { };
  private onTouched: () => void = (): void => { };

  private readonly docConversionSrv = inject(DocumentConversionService);
  private readonly informerSrv = inject(InformerService);

  @HostListener('click', ['$event'])
  onClick(): void {
    if (!this.value()) {
      this.fileInput().nativeElement?.click();
    }
  }

  maxImageMb(): string { return String(Math.round(this.maxImageSize() / (1024 * 1024))); }
  maxPdfMb(): string { return String(Math.round(this.maxPdfSize() / (1024 * 1024))); }

  sizeText(): string {
    const v = this.value();
    if (!v) { return ''; }
    const b = v.size;
    if (b < 1024) { return `${b} B`; }
    if (b < 1024 * 1024) { return `${(b / 1024).toFixed(1)} KB`; }
    return `${(b / (1024 * 1024)).toFixed(2)} MB`;
  }

  allowedMb(): number {
    const v = this.value();
    if (!v) { return this.maxImageSize() / (1024 * 1024); }
    return v.type === UserDocumentType.Pdf ? (this.maxPdfSize() / (1024 * 1024)) : (this.maxImageSize() / (1024 * 1024));
  }

  writeValue(obj: DocumentValue | null): void {
    this.value.set(obj);
  }
  registerOnChange(fn: (v: DocumentValue | null) => void): void { this.onChange = fn; }
  registerOnTouched(fn: () => void): void { this.onTouched = fn; }
  setDisabledState(isDisabled: boolean): void { this.disabled.set(isDisabled); }

  async onFile(event: Event): Promise<void> {
    const inputEl = event.target as HTMLInputElement | null;
    if (!inputEl || !inputEl.files || inputEl.files.length === 0) { return; }
    const f = inputEl.files[0];
    inputEl.value = '';
    await this.handleFileUpload(f);
    this.onTouched();
  }

  private async handleFileUpload(file: File | null): Promise<void> {
    this.errors.set(null);
    try {
      if (!file) {
        throw new Error('Файл не обнаружен');
      }

      const isImage = this.docConversionSrv.isImage(file);
      const isPdf = this.docConversionSrv.isPdf(file);

      if (!isImage && !isPdf) {
        throw new Error('Некорректный формат файла');
      }

      const tooLarge = isPdf ? file.size > this.maxPdfSize() : file.size > this.maxImageSize();
      if (tooLarge) {
        throw new Error('Файл слишком тяжелый');
      }

      const doc = await this.docConversionSrv.convertFileToDataUrl(file);
      if (!doc) {
        throw new Error('Не удалось обработать файл');
      }

      this.setValue(doc);
    } catch (err) {
      this.informerSrv.error(err);
      this.errors.set({ convertFailed: true });
      this.setValue(null);
    }
  }

  private setValue(v: DocumentValue | null): void {
    this.value.set(v);
    this.valueChange.emit(v);
    this.onChange(v);
  }

  clear(): void {
    setTimeout(() => {
      this.setValue(null);
      this.errors.set(null);
    })
  }

  validate(_: AbstractControl): ValidationErrors | null { return this.errors(); }

}
