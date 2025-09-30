import { ChangeDetectionStrategy, Component, inject} from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { DocumentValue } from '../../services/document-conversion.service';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { DragDropModule } from '@angular/cdk/drag-drop';

@Component({
  selector: 'app-document-preview-modal',
  templateUrl: './document-preview-modal.component.html',
  styleUrl: './document-preview-modal.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  imports: [MatIconModule, MatButtonModule, DragDropModule],
})
export class DocumentPreviewModalComponent {
  readonly dialogRef = inject(MatDialogRef<DocumentPreviewModalComponent>, { optional: true });
  readonly data = inject<DocumentValue | null>(MAT_DIALOG_DATA, { optional: true });

  close(): void {
    this.dialogRef?.close();
  }
}
