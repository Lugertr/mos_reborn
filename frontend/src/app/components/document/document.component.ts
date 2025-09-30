import { CommonModule, NgOptimizedImage } from '@angular/common';
import { Component, ChangeDetectionStrategy, computed, signal, inject, effect, Signal } from '@angular/core';
import { FormArray, FormBuilder, FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { DocumentService } from './services/document.service';
import { DocumentUploaderComponent, FileData } from './document-uploader/document-uploader.component';
import { DocumentImageComponent } from './document-image/document-image.component';
import { OcrBackendResponse } from './models/types';
import { isFilledArray } from '@core/helpers/array-utils';
import { isNumber } from '@core/helpers/number-utils';
import { map } from 'rxjs';

export interface DocumentForm {
  documentTitle: FormControl<string>;
  file: FormControl<string>;
  fields: FormArray<FormGroup<DocumentsField>>
}

export interface DocumentsField {
  coords: FormControl<[number, number]>,
  strValue: FormControl<string>,
  strOcr: FormControl<string>,
}

@Component({
  selector: 'app-document',
  standalone: true,
  imports: [
    CommonModule, ReactiveFormsModule,
    MatButtonModule, MatFormFieldModule, MatInputModule, MatListModule, MatIconModule,
    DocumentUploaderComponent, DocumentImageComponent,
  ],
  templateUrl: './document.component.html',
  styleUrls: ['./document.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DocumentComponent {
  private readonly docSrv = inject(DocumentService);

  readonly form = new FormArray<FormGroup<DocumentForm>>([])
  readonly selectedIndex = signal<number>(0);

  hasDocuments: Signal<boolean> = toSignal(this.form.valueChanges.pipe(map(val => !!val?.length)));

  readonly selectedDocument = computed<FormGroup<DocumentForm>>(() => {
    const i = this.selectedIndex();
    return this.documents.at(i) ?? null;
  });

  get documents(): FormGroup<DocumentForm>[] {
    return this.form.controls;
  }

  onAddDocument(): void {
    const i = this.documents.length + 1;
    this.addDocument('', `Документ ${i}`, []);
  }

  onFilesSelected(files: FileData[]) {
    files.forEach((f, idx) => this.addDocument(f.dataUrl, f.name || `Документ ${this.documents.length + idx + 1}`, []));
    if (this.documents.length > 0) this.selectedIndex.set(0);
    console.log(this.form);
  }

  /** Select a document in the list */
  onSelectDocument(idx: number): void {
    this.selectedIndex.set(idx);
  }

  /** Remove a document */
  removeDocument(idx: number): void {
    if (idx < 0 || idx >= this.documents.length) return;
    this.form.removeAt(idx);
    if (this.documents.length === 0) {
      this.selectedIndex.set(0);
    } else if (this.selectedIndex() >= this.documents.length) {
      this.selectedIndex.set(this.documents.length - 1);
    }
  }

  onGenerate(): void {
    const group = this.selectedDocument();
    const file = group?.value?.file;
    if (!file) return;

    this.docSrv.predict(file)
      .pipe(takeUntilDestroyed())
      .subscribe({
        next: (resp: OcrBackendResponse) => {
          const fieldsArray = new FormArray<FormGroup<DocumentsField>>([]);

          for (const item of resp) {
            if (item || !isFilledArray(item.coords)) {
              continue;
            }
            const [x, y] = item.coords;
            fieldsArray.push(new FormGroup({
              coords: new FormControl<[number, number]>([x || 0, y || 0]),
              strValue: new FormControl<string>(item.value),
              strOcr: new FormControl<string>(item.value),
            }));
          }
        },
        error: (err) => {
          console.error('OCR request failed', err);
        }
      });
  }

  private addDocument(base64: string, name: string, fields: Array<{ x: number; y: number; strValue: string; strOcr: string }>) {

    const documentForm = new FormGroup({
      documentTitle: new FormControl<string>(name),
      file: new FormControl<string>(base64),
      fields: new FormArray<FormGroup<DocumentsField>>([])
    });

    for (const f of fields) {
      if (!isNumber(!f.x) && isNumber(!f.y)) {
        continue;
      }
      documentForm.controls.fields.push(new FormGroup({
        coords: new FormControl<[number, number]>([f.x, f.y]),
        strValue: new FormControl<string>(f.strValue),
        strOcr: new FormControl<string>(f.strOcr),
      }));
    }

    this.form.push(documentForm);
  }
}
