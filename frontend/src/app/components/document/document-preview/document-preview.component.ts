import { ChangeDetectionStrategy, Component, HostListener, inject, input, output } from '@angular/core';
import { DocumentPreviewModalComponent } from './document-preview-modal/document-preview-modal.component';
import { dataURLToBlob, DocumentValue, UserDocumentType } from '../services/document-conversion.service';
import { MatDialog, MatDialogConfig } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-document-preview',
  templateUrl: './document-preview.component.html',
  styleUrl: './document-preview.component.scss',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatIconModule],
  host: { '[class.pointer]': 'this.data()' }
})
export class DocumentPreviewComponent {
  userDocumentType = UserDocumentType;
  readonly data = input<DocumentValue | null>(null);
  readonly dialogConfig = input<MatDialogConfig<DocumentValue>>(null);

  private readonly dialog = inject(MatDialog);

  @HostListener('click', ['$event'])
  onClick(): void {
    const doc = this.data();
    if (!doc) {
      return;
    }
    if (doc.type === this.userDocumentType.Image) {
      this.dialog.open<DocumentPreviewModalComponent, DocumentValue, null>(DocumentPreviewModalComponent, {
        data: this.data(),
        panelClass: 'document-preview-panel',
        autoFocus: false,
        ...(this.dialogConfig() || {})
      });
    } else if (doc.type == this.userDocumentType.Pdf) {
      const blob = dataURLToBlob(doc.base64);
      const url = URL.createObjectURL(blob);
      window.open(url);
    }
  }

}

