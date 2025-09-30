import { ChangeDetectionStrategy, Component } from '@angular/core';
import { Observable } from 'rxjs';
import { delay } from 'rxjs/operators';
import { LoadingBarService } from './loading-bar.service';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { AsyncPipe } from '@angular/common';

@Component({
  selector: 'app-loading-bar',
  template: `
    @if (show$ | async) {
      <mat-progress-bar mode="indeterminate" color="primary"></mat-progress-bar>
    }
  `,
  styles: [
    `
      :host {
        display: block;
        position: absolute;
        width: 100%;
        top: 0;
        left: 0;
        z-index: 10000;
      }
    `,
  ],
  standalone: true,
  imports: [MatProgressBarModule, AsyncPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class LoadingBarComponent {
  show$: Observable<boolean>;
  constructor(private loadingBarSrv: LoadingBarService) {
    this.show$ = this.loadingBarSrv.show$.pipe(delay(0));
  }
}
