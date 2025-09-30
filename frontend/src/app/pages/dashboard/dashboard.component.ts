import { ChangeDetectionStrategy, Component, DestroyRef, inject, OnInit, signal, ViewContainerRef } from '@angular/core';
import { ArchiveDocument, DashboardService } from './services/dashboard.service';
import { catchError, filter, mergeMap, of, Subject } from 'rxjs';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { LoadingBarService } from '@core/loading-bar/loading-bar.service';
import { InformerService } from '@core/services/informer.service';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { DashboardTableComponent } from './dashboard-table/dashboard-table.component';
import { MatDialog } from '@angular/material/dialog';
import { DashboardCreateComponent } from './dashboard-create/dashboard-create.component';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { DashboardCardComponent } from './dashboard-card/dashboard-card.component';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
  standalone: true,
  imports: [MatButtonModule, MatIconModule, MatCardModule, DashboardTableComponent, MatFormFieldModule, DashboardCardComponent, MatInputModule],
  providers: [DashboardService],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DashboardComponent implements OnInit {
  private readonly documentService = inject(DashboardService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly informerSrv = inject(InformerService);
  private readonly loadingBarSrv = inject(LoadingBarService);

  private dialog = inject(MatDialog);
  private readonly vc = inject(ViewContainerRef);

  readonly isLoading = toSignal(this.loadingBarSrv.show$);

  private readonly reload$ = new Subject<void>();
  readonly documents = signal<ArchiveDocument[]>([]);
  readonly filter = signal('');
  readonly selectedDoc = signal<ArchiveDocument | null>(null);

  constructor() {
    this.reload$.pipe(
      mergeMap(() => this.documentService.getALLDocuments().pipe(this.loadingBarSrv.withLoading(), catchError((err) => {
        this.informerSrv.error(err?.error?.message, 'Ошибка получения документов');
        return of(null);
      }))),
      takeUntilDestroyed(this.destroyRef)
    ).subscribe((documents) => {
      this.documents.set(documents || []);
      console.log(documents);
    })
  }

  ngOnInit(): void {
    this.refresh();
  }

  refresh(): void {
    this.reload$.next();
  }

  selectDocument(document: ArchiveDocument): void {

    if (!document) {
      this.selectedDoc.set(null);
      return;
    }
    this.documentService.getDocumentById(document.doc_id).pipe(
      this.loadingBarSrv.withLoading(),
      takeUntilDestroyed(this.destroyRef)
    ).subscribe({
      next: (doc) => {
        if (doc) {
          this.selectedDoc.set(doc);
        }
      },
      error: (err) => {
        this.selectedDoc.set(null);
        this.informerSrv.error(err?.error?.message, 'Ошибка получения информации о документе')
      }
    })

  }

  addDocument(): void {
    this.dialog.open<DashboardCreateComponent, null, void>(DashboardCreateComponent, {
      data: null,
      disableClose: true,
    })
      .afterClosed()
      .pipe(filter(Boolean),
        mergeMap(() => this.documentService.getALLDocuments().pipe(this.loadingBarSrv.withLoading())),
        takeUntilDestroyed(this.destroyRef)
      ).subscribe({
        next: () => {
          this.informerSrv.success('Документ успешно добавлен');
          this.refresh();
        },
        error: (err) => {
          this.informerSrv.error(err?.error?.message, 'Ошибка создания документа');
        }
      })
  }

  editDocument(document: ArchiveDocument): void {
    this.dialog.open<DashboardCreateComponent, ArchiveDocument, void>(DashboardCreateComponent, {
      data: document,
      disableClose: true,
    })
      .afterClosed()
      .pipe(filter(Boolean),
        mergeMap(() => this.documentService.getALLDocuments().pipe(this.loadingBarSrv.withLoading())),
        takeUntilDestroyed(this.destroyRef)
      ).subscribe({
        next: () => {
          this.informerSrv.success('Документ успешно изменен');
          this.refresh();
        },
        error: (err) => {
          this.informerSrv.error(err?.error?.message, 'Ошибка редактирования документа');
        }
      })
  }


}
