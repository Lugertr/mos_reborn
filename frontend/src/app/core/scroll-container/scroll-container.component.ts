import { ScrollingModule } from '@angular/cdk/scrolling';
import { ChangeDetectionStrategy, Component, ElementRef, EventEmitter, HostBinding, Output, ViewChild } from '@angular/core';
import { ScrollContainerDirective } from './scroll-container.directive';

@Component({
  selector: 'app-scroll-container',
  template: '<div #scrollBox class="scroll-box" cdk-scrollable (scrolledBottom)="scrolledBottom.emit()"><ng-content></ng-content></div>',
  styleUrls: ['./scroll-container.component.scss'],
  imports: [ScrollingModule, ScrollContainerDirective],
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
})
export class ScrollContainerComponent {
  @Output() scrolledBottom = new EventEmitter<void>();
  @HostBinding('class.scroll-container') cssClass = true;
  @ViewChild('scrollBox') private scrollBox: ElementRef<HTMLDivElement>;

  scrollTop(): void {
    if (this.scrollBox && this.scrollBox.nativeElement) {
      const el: HTMLElement = this.scrollBox.nativeElement;
      el.scrollTop = 0;
    }
  }
}
