import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  input,
  Output,
} from '@angular/core';
import type { WritableSignal } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatTooltipModule } from '@angular/material/tooltip';
import { PrivacyLabelPipe } from '../pipes/document-privacy.pipe';

export type PrivacyType = 'public' | 'private';

export interface ArchiveDocument {
  doc_id: number;
  title: string;
  privacy: PrivacyType;
  created_at?: string | null;
  document_date?: string | null;
  author?: string | null;
  type_id?: number | null;
  can_requester_edit?: boolean;
}

/**
 * Standalone document card.
 * Принимает сигнал: [doc]="docSignal"
 * Эмитит события: (edit), (delete), (changePrivacy)
 */
@Component({
  selector: 'app-dashboard-card',
  standalone: true,
  imports: [CommonModule, MatCardModule, MatIconModule, MatButtonModule, MatTooltipModule, PrivacyLabelPipe],
  templateUrl: './dashboard-card.component.html',
  styleUrls: ['./dashboard-card.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardCardComponent {
  doc = input<ArchiveDocument | null>(null);

  @Output() edit = new EventEmitter<ArchiveDocument>();
  @Output() remove = new EventEmitter<ArchiveDocument>();
  @Output() changePrivacy = new EventEmitter<{ doc: ArchiveDocument; newPrivacy: PrivacyType }>();

  formatDate(date?: string | null): string {
    if (!date) return '-';
    // отображаем только дату в формате yyyy-MM-dd
    try {
      const d = new Date(date);
      if (isNaN(d.getTime())) return '-';
      // простой формат — yyyy-MM-dd
      const y = d.getUTCFullYear();
      const m = String(d.getUTCMonth() + 1).padStart(2, '0');
      const day = String(d.getUTCDate()).padStart(2, '0');
      return `${y}-${m}-${day}`;
    } catch {
      return '-';
    }
  }

  onEdit() {
    const d = this.doc();
    if (d && d.can_requester_edit) this.edit.emit(d);
  }
  onRemove() {
    const d = this.doc();
    if (d && d.can_requester_edit) this.remove.emit(d);
  }
  onTogglePrivacy() {
    const d = this.doc();
    if (!d) return;
    const newPrivacy: PrivacyType = d.privacy === 'public' ? 'private' : 'public';
    this.changePrivacy.emit({ doc: d, newPrivacy });
  }
}
