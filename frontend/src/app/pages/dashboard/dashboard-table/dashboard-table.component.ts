import { AfterViewInit, ChangeDetectionStrategy, Component, effect, EventEmitter, input, output, Output, signal, ViewChild } from '@angular/core';
import { MatTableModule } from '@angular/material/table';
import { TableDataSource } from '@core/table-data-source/table-data-source.class';
import { ScrollContainerComponent } from '@core/scroll-container/scroll-container.component';
import { DatePipe } from '@angular/common';
import { ArchiveDocument, PrivacyType } from '../services/dashboard.service';
import { MatSort, MatSortModule } from '@angular/material/sort';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatIconModule } from '@angular/material/icon';


@Component({
  selector: 'app-dashboard-table',
  templateUrl: './dashboard-table.component.html',
  styleUrl: './dashboard-table.component.scss',
  standalone: true,
  imports: [DatePipe, MatTableModule, MatSortModule, ScrollContainerComponent, MatTooltipModule, MatIconModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardTableComponent implements AfterViewInit {
  @ViewChild(MatSort) sort: MatSort;
  rowSelected = output<ArchiveDocument>();
  documents = input<ArchiveDocument[]>([]);
  filter = input<string>('');
  readonly selected = signal<ArchiveDocument | null>(null);

  readonly dataSource = new TableDataSource<ArchiveDocument>([]);
  readonly displayedColumns = ['doc_id', 'title', 'type_name', 'author', 'document_date', 'privacy'];

  constructor() {
    effect(() => {
      this.dataSource.data = this.documents();
    })

    effect(() => {
      const f = (this.filter() ?? '').trim().toLowerCase();
      this.dataSource.filter = f;
    });
  }

  ngAfterViewInit(): void {
    this.dataSource.sort = this.sort;
  }

  onRowClick(row: ArchiveDocument): void {
    const cur = this.selected();
    if (cur && cur.doc_id === row.doc_id) {
      this.selected.set(null);
      this.rowSelected.emit(null);
    } else {
      this.selected.set(row);
      this.rowSelected.emit(row);
    }
  }

  isPublicDocument(docStatus: PrivacyType): boolean {
    return docStatus === PrivacyType.Public;
  }
}
