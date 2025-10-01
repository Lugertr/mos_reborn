import { CommonModule } from '@angular/common';
import { Component, ChangeDetectionStrategy, computed, signal, inject, Signal, DestroyRef, ChangeDetectorRef, OnInit } from '@angular/core';
import { FormArray, FormControl, FormGroup, ReactiveFormsModule, } from '@angular/forms';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { DocumentService } from './services/document.service';
import { DocumentUploaderComponent } from './document-uploader/document-uploader.component';
import { DocumentImageComponent } from './document-image/document-image.component';
import { isFilledArray } from '@core/helpers/array-utils';
import { isNumber } from '@core/helpers/number-utils';
import { map } from 'rxjs';
import { InformerService } from '@core/services/informer.service';
import { FileConversionService, FileData } from './services/file-conversion.service';
import { LoadingBarService } from '@core/loading-bar/loading-bar.service';

export interface DocumentForm {
  documentTitle: FormControl<string>;
  file: FormControl<File>;
  fields: FormArray<FormGroup<DocumentsFieldForm>>;
  size: FormControl<[number, number]>;
}

export interface DocumentsFieldForm {
  coords: FormControl<[number, number]>,
  strValue: FormControl<string>,
  strOcr: FormControl<string>,
  state: FormControl<FieldFormState>,
  width: FormControl<number>,
  height: FormControl<number>
}

export enum FieldFormState {
  Default,
  Hidden,
  Moving
}

@Component({
  selector: 'app-document',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    MatButtonModule, MatFormFieldModule, MatInputModule, MatListModule, MatIconModule,
    DocumentUploaderComponent, DocumentImageComponent,
  ],
  templateUrl: './document.component.html',
  styleUrls: ['./document.component.scss'],
  providers: [FileConversionService],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DocumentComponent implements OnInit {
  private readonly docSrv = inject(DocumentService);
  private readonly informer = inject(InformerService);
  private readonly loadingBarSrv = inject(LoadingBarService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly fileConversion = inject(FileConversionService);
  private readonly cdr = inject(ChangeDetectorRef);

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

  ngOnInit(): void {
    this.docSrv.serverConnection().pipe(
      this.loadingBarSrv.withLoading(),
      takeUntilDestroyed(this.destroyRef)).subscribe(({
        error: (err) => {
          this.informer.error(err, 'Нет подключения к серверу');
        }
      }))
  }

  async onAddDocument(ev: Event): Promise<void> {
    const files = await this.fileConversion.onSelectFiles(ev);
    files.forEach((f) => this.addDocument(f, []));
    this.cdr.markForCheck();
  }

  onFilesSelected(files: FileData[]) {
    files.forEach((f) => this.addDocument(f, []));
    if (this.documents.length > 0) this.selectedIndex.set(0);
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
    this.cdr.markForCheck();

  }

  test(): void {
        this.docSrv.test().pipe(takeUntilDestroyed(this.destroyRef)).subscribe()
  }

  onGenerate(): void {
    const doc = this.selectedDocument();
    const file = doc?.getRawValue()?.file;
    if (!file) return;

    const docData = doc.getRawValue();
    this.docSrv.predict({
      file: file,
      img_width: `${docData.size[1]}`,
      img_height: `${docData.size[0]}`
    })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: resp => {
          const docFields: FormGroup<DocumentsFieldForm>[] = [];
          for (const item of resp) {
            if (!item || !isFilledArray(item.coords)) {
              continue;
            }
            const [x, y] = item.coords;

            docFields.push(new FormGroup({
              coords: new FormControl<[number, number]>([x || 0, y || 0]),
              strValue: new FormControl<string>(item.value),
              strOcr: new FormControl<string>(item.value),
              state: new FormControl<FieldFormState>(FieldFormState.Default),
              width: new FormControl(item.width),
              height: new FormControl(item.height)
            }));

          }
          doc.setControl('fields', new FormArray<FormGroup<DocumentsFieldForm>>(docFields));
        },
        error: (err) => {
          this.informer.error(err, 'Не удалось проанализировать документ');
        }
      });
  }

  private addDocument(file: FileData, fields: Array<{ x: number; y: number; strValue: string; strOcr: string }>) {

    const documentForm = new FormGroup({
      documentTitle: new FormControl(file.name),
      file: new FormControl(file.file),
      fields: new FormArray<FormGroup<DocumentsFieldForm>>([]),
      size: new FormControl({ disabled: true, value: file.size })
    });
    for (const f of fields) {
      if (!isNumber(!f.x) && isNumber(!f.y)) {
        continue;
      }
      documentForm.controls.fields.push(new FormGroup({
        coords: new FormControl<[number, number]>([f.x, f.y]),
        strValue: new FormControl<string>(f.strValue),
        strOcr: new FormControl<string>(f.strOcr),
        state: new FormControl<FieldFormState>(FieldFormState.Default),
        width: new FormControl(null),
        height: new FormControl(null)
      }));
    }
    this.form.push(documentForm);
  }
}
