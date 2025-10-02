import { Component, ChangeDetectionStrategy, signal, inject, Signal, DestroyRef, ChangeDetectorRef, OnInit } from '@angular/core';
import { FormArray, FormControl, FormGroup, ReactiveFormsModule, } from '@angular/forms';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatListModule } from '@angular/material/list';
import { MatIconModule } from '@angular/material/icon';
import { DocumentService, SegmentOut } from './services/document.service';
import { DocumentUploaderComponent } from './document-uploader/document-uploader.component';
import { DocumentImageComponent } from './document-image/document-image.component';
import { isFilledArray } from '@core/helpers/array-utils';
import { isNumber } from '@core/helpers/number-utils';
import { filter, map, tap } from 'rxjs';
import { InformerService } from '@core/services/informer.service';
import { FileConversionService, FileData } from './services/file-conversion.service';
import { LoadingBarService } from '@core/loading-bar/loading-bar.service';
import { OcrPhase, ProgressStatus, resolveProgress, STATUS_MAP } from '@core/helpers/progress';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import { buildCsvFromFormArray, downloadCsv } from '@core/helpers/csv-builder';

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
  height: FormControl<number>,
  isSelected: FormControl<boolean>
}

export enum FieldFormState {
  Default,
  Hidden,
  Moving
}

interface CsvExportData {
  ind: number;
  title: string;
  strValue: string,
  strOcr: string;
  coords: [number, number],
}

@Component({
  selector: 'app-document',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    MatButtonModule, MatFormFieldModule, MatInputModule, MatListModule, MatIconModule,
    DocumentUploaderComponent, DocumentImageComponent, MatProgressBarModule, MatProgressSpinnerModule,
    MatTooltipModule
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
  readonly generatedCount = signal<number>(0);

  readonly fileProgressMap = new Map<number, ProgressStatus>();
  readonly isLoading: Signal<boolean> = toSignal(this.loadingBarSrv.show$);
  hasDocuments: Signal<boolean> = toSignal(this.form.valueChanges.pipe(map(val => !!val?.length)));
  readonly fieldFormState = FieldFormState;


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

  get selectedDocument(): FormGroup<DocumentForm> {
    const i = this.selectedIndex();
    return this.documents.at(i) ?? null;
  }

  getSelectedLoadingStatus(ind: number = this.selectedIndex()): ProgressStatus {
    return this.fileProgressMap.get(ind);
  }

  async onAddDocument(ev: Event): Promise<void> {
    const files = await this.fileConversion.onSelectFiles(ev);
    files.forEach((f) => this.addDocument(f, []));
    this.cdr.markForCheck();
  }

  onFilesSelected(files: FileData[]) {
    files.forEach((f) => this.addDocument(f, []));
    if (this.documents.length > 0) this.selectedIndex.set(0);
    this.cdr.detectChanges();
  }

  onSelectDocument(idx: number): void {
    this.selectedIndex.set(idx);
  }

  downloadCsv(): void {
    const docs = this.form.getRawValue();

    const raws: CsvExportData[] = [];
    for (const doc of docs) {
      for (let i = 0; i < doc.fields.length; i++) {
        raws.push({
          ind: i,
          title: doc.documentTitle,
          strValue: doc.fields[i].strValue,
          strOcr: doc.fields[i].strOcr,
          coords: doc.fields[i].coords,
        });
      }
    }

    const csv = buildCsvFromFormArray(raws,
      ['ind','title','coords[0]', 'coords[1]', 'strValue', 'strOcr'],
      ['id','Название документа','x', 'y', 'значение', 'предварительное значение']
    );

    downloadCsv(csv, 'document_fields.csv');

  }

  removeDocument(idx: number): void {
    if (idx < 0 || idx >= this.documents.length) return;
    this.fileProgressMap.delete(idx);
    this.form.removeAt(idx);
    if (this.documents.length === 0) {
      this.selectedIndex.set(0);
    } else if (this.selectedIndex() >= this.documents.length) {
      this.selectedIndex.set(this.documents.length - 1);
    }
    this.cdr.markForCheck();

  }

  onSelectSegment(seg: FormGroup<DocumentsFieldForm>, arr: FormGroup<DocumentsFieldForm>[]): void {
    arr.forEach(arrSeg => arrSeg.patchValue({ isSelected: arrSeg === seg }));
  }

  test(): void {
    this.docSrv.test().pipe(takeUntilDestroyed(this.destroyRef)).subscribe()
  }

  deleteSegment(array: FormArray<FormGroup<DocumentsFieldForm>>, ind: number): void {
    array.removeAt(ind);
    this.cdr.markForCheck();
  }

  onGenerate(): void {
    const doc = this.selectedDocument;
    const file = doc?.getRawValue()?.file;
    if (!file) return;
    const selectedInd = this.selectedIndex();
    this.fileProgressMap.set(selectedInd, STATUS_MAP.get(OcrPhase.Start))

    const docData = doc.getRawValue();
    this.docSrv.predictEvents({
      file: file,
      img_width: `${docData.size[1]}`,
      img_height: `${docData.size[0]}`
    })
      .pipe(
        tap(ev => {
          if (ev.kind !== 'progress') return;
          const s = resolveProgress(ev.data);
          this.fileProgressMap.set(selectedInd, s);
          this.cdr.markForCheck();
        }),
        filter(ev => ev?.kind === 'result'),
        map(ev => (ev as { kind: 'result', data: { segments: SegmentOut[] } }).data.segments),
        this.loadingBarSrv.withLoading(),
        takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: resp => {
          this.generatedCount.update(v => v + 1);
          const docFields: FormGroup<DocumentsFieldForm>[] = [];
          for (const item of resp) {
            if (!item || !isFilledArray(item.coords)) {
              continue;
            }
            const [x, y] = item.coords;

            docFields.push(new FormGroup({
              coords: new FormControl<[number, number]>([x || 0, y || 0]),
              strValue: new FormControl<string>(item.value),
              strOcr: new FormControl<string>(item.preview_value),
              state: new FormControl<FieldFormState>(FieldFormState.Default),
              width: new FormControl(item.width),
              height: new FormControl(item.height),
              isSelected: new FormControl(false)
            }));

          }
          const newFormArr = new FormArray<FormGroup<DocumentsFieldForm>>(docFields);
          doc.setControl('fields', newFormArr);
        },
        error: (err) => {
          this.informer.error(err, 'Не удалось проанализировать документ');
        }
      });
  }

  swapSegmentValue(seg: FormGroup<DocumentsFieldForm>): void {
    const str = seg.controls.strValue.getRawValue();
    seg.controls.strValue.setValue(seg.controls.strOcr.getRawValue());
    seg.controls.strOcr.setValue(str);
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
        height: new FormControl(null),
        isSelected: new FormControl(false)

      }));
    }
    this.form.push(documentForm);
  }
}
