import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { defer, Observable } from 'rxjs';
import { finalize } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class LoadingBarService {
  private loaderSubject = new BehaviorSubject<boolean>(false);
  private active = 0;

  show$ = this.loaderSubject.asObservable();

  show(): void {
    this.active++;
    this.loaderSubject.next(true);
  }

  hide(): void {
    this.active--;
    if (this.active < 1) {
      this.loaderSubject.next(false);
    }
  }

  withLoading<T>(): (src$: Observable<T>) => Observable<T> {
    return (src$: Observable<T>) =>
      defer(() => {
        this.show();
        return src$.pipe(
          finalize(() => {
            this.hide();
          }),
        );
      });
  }
}
