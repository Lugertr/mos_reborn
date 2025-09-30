import { DestroyRef, Directive, ElementRef, EventEmitter, NgZone, OnDestroy, OnInit, Output, Self } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { fromEvent, throttleTime } from 'rxjs';

const TOLERANCE = 1;

@Directive({
  selector: '[scrolledBottom]',
  standalone: true,
})
export class ScrollContainerDirective implements OnInit, OnDestroy {
  @Output('scrolledBottom')
  readonly scrolledBottom = new EventEmitter<Event>();
  readonly viewportEl: Element;

  constructor(
    @Self() viewport: ElementRef,
    private readonly zone: NgZone,
    @Self() protected readonly destroy$: DestroyRef,
  ) {
    this.viewportEl = viewport.nativeElement;
  }

  ngOnInit(): void {
    this.zone.runOutsideAngular(() => {
      fromEvent(this.viewportEl, 'scroll', { passive: true })
        .pipe(
          throttleTime(250, undefined, {
            leading: false,
            trailing: true,
          }),
          takeUntilDestroyed(this.destroy$),
        )
        .subscribe(e => {
          this.zone.runTask(() => this.onScroll(e));
        });
    });
  }

  ngOnDestroy(): void {
    this.scrolledBottom.complete();
  }

  onScroll(e: Event): void {
    if (this.viewportEl.scrollHeight - this.viewportEl.scrollTop - this.viewportEl.clientHeight < TOLERANCE) {
      this.scrolledBottom.emit(e);
    }
  }
}
