import { ChangeDetectionStrategy, Component, signal } from '@angular/core';
import { DocumentReaderComponent } from './document-reader/document-reader.component';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { DocumentValue } from './services/document-conversion.service';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { DocumentFormComponent, DocumentFormGroup } from './document-form/document-form.component';
import { DocumentPreviewComponent } from './document-preview/document-preview.component';

enum DocumentStage {
  FileReader,
  FileForm
}

@Component({
  selector: 'app-document1',
  templateUrl: './document1.component.html',
  styleUrl: './document1.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  imports: [DocumentReaderComponent, MatButtonModule, MatIconModule, DocumentFormComponent, DocumentPreviewComponent, MatTooltipModule, ReactiveFormsModule]
})
export class DocumentComponent {
  form = new FormGroup({
    file: new FormControl<DocumentValue>(null),
    doc: new FormGroup<DocumentFormGroup>({
      title: new FormControl<string | null>(null, [Validators.required]),
      isPublic: new FormControl(false, { nonNullable: true, validators: [Validators.required] }),
      isNeedSave: new FormControl(true, { nonNullable: true, validators: [Validators.required] }),
      documentDate: new FormControl<Date | null>(null, [Validators.required]),
      authorName: new FormControl<string | null>(null, [Validators.required]),
      typeId: new FormControl<number | null>(null, [Validators.required]),
      tags: new FormControl<string[]>([]),
      geojsonText: new FormControl<string | null>(null, [Validators.required]),
    })
  });

  readonly documentStage = DocumentStage;
  readonly stage = signal(DocumentStage.FileReader);

  get isFileReader(): boolean {
    return this.stage() === DocumentStage.FileReader;
  }

  nexStage(stage: DocumentStage): void {
    this.stage.set(stage);
  }
}
