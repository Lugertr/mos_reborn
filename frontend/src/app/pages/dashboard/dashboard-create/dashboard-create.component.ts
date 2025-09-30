import { ChangeDetectionStrategy, Component, computed, inject, Inject, signal } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA, MatDialogModule } from '@angular/material/dialog';
import { FormBuilder, FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatDatepickerModule } from '@angular/material/datepicker';
import { MatChipInputEvent, MatChipsModule } from '@angular/material/chips';
import { ArchiveDocument, PrivacyType } from '../services/dashboard.service';
import { provideNativeDateAdapter } from '@angular/material/core';
import { LoadingBarService } from '@core/loading-bar/loading-bar.service';
import { toSignal } from '@angular/core/rxjs-interop';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { FileDropComponent } from '@core/file-drop/file-drop.component';
import {MatExpansionModule} from '@angular/material/expansion';
import { TextFieldModule } from '@angular/cdk/text-field';

@Component({
  selector: 'app-dashboard-create',
  templateUrl: './dashboard-create.component.html',
  styleUrl: './dashboard-create.component.scss',
  standalone: true,
  providers: [provideNativeDateAdapter()],
  imports: [
    ReactiveFormsModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatSlideToggleModule,
    MatSelectModule,
    MatDatepickerModule,
    MatButtonModule,
    MatIconModule,
    MatChipsModule,
    FileDropComponent,
    MatExpansionModule,
    TextFieldModule
  ],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardCreateComponent {
  private readonly dialogRef = inject(MatDialogRef<DashboardCreateComponent, null>);
  private readonly loadingBarSrv = inject(LoadingBarService);

  readonly isLoading = toSignal(this.loadingBarSrv.show$);
  private readonly _file = signal<File | null>(null);
  readonly privacyEnum = PrivacyType;

  readonly selectedFileName = computed(() => {
    const f = this._file();
    return f ? f.name : 'Файл не выбран';
  });

  readonly form = new FormGroup({
    title: new FormControl<string | null>(null, { validators: [Validators.required] }),
    isPublic: new FormControl(false, { nonNullable: true }),
    isNeedSave: new FormControl(true, { nonNullable: true }),
    documentDate: new FormControl<Date | null>(null),
    authorName: new FormControl<string | null>(null),
    typeId: new FormControl<number | null>(null),
    tags: new FormControl<string[]>([]),
    geojsonText: new FormControl<string | null>(null),
  });


  constructor(
    @Inject(MAT_DIALOG_DATA) public data: ArchiveDocument,
  ) {
    if (data) {
      this.form.setValue({
        title: data.title ?? null,
        isPublic: false,
        documentDate: data.document_date ? new Date(data.document_date) : null,
        authorName: data.author_name ?? null,
        isNeedSave: true,
        typeId: data.type_id ?? null,
        tags: data.tags || [],
        geojsonText: null,
      });
    }
  }


  onFile(ev: Event): void {
    const input = ev.target as HTMLInputElement;
    const file = input.files && input.files.length > 0 ? input.files[0] : null;
    this._file.set(file);
  }

  onFilesChanged(files: File[]): void {
    console.log(files);
  }

  removeTag(keyword: string): void {
    this.form.controls.tags.setValue(this.form.controls.tags.getRawValue().filter(val => val !== keyword));
  }

  addTag(event: MatChipInputEvent): void {
    const value = (event.value || '').trim();
    if (value) {
      this.form.controls.tags.setValue([...this.form.controls.tags.getRawValue(), value])
    }
    event.chipInput!.clear();
  }

  submit(): void {
    console.log(this.form.getRawValue());
    const raw = this.form.getRawValue();
    let geojson: unknown | null = null;
  }
}

