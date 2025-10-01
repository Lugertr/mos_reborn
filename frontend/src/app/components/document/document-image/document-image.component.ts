import {
  Component,
  ChangeDetectionStrategy,
  input,
  computed,
  inject,
  Signal,
  DestroyRef,
  effect,
  ChangeDetectorRef,
} from '@angular/core';
import { FormGroup, ReactiveFormsModule } from '@angular/forms';
import { DocumentForm, DocumentsFieldForm, FieldFormState } from '../document.component';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { Subject, Subscription } from 'rxjs';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DragDropModule } from '@angular/cdk/drag-drop';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import {MatMenuModule} from '@angular/material/menu';

/**
 * Displays the image and overlays form inputs at coordinate positions.
 * The coordinates are interpreted as CSS pixels relative to the natural image size.
 */
@Component({
  selector: 'app-document-image',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    DragDropModule,
    MatTooltipModule,
    MatIconModule,
    MatButtonModule,
    MatMenuModule,
  ],
  templateUrl: './document-image.component.html',
  styleUrls: ['./document-image.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DocumentImageComponent {
  private sanitizer = inject(DomSanitizer);
  private destroyRef = inject(DestroyRef);
  private cdr = inject(ChangeDetectorRef);

  readonly fieldFormState = FieldFormState;
  readonly data = input.required<FormGroup<DocumentForm>>();

  previewImg: Signal<SafeUrl> = computed(() => {
    const imgFile = this.data()?.value?.file;
    if (imgFile) {
      const objUrl = URL.createObjectURL(imgFile);
      return this.sanitizer.bypassSecurityTrustUrl(objUrl);
    }
    return null;
  });

  reload$ = new Subject<void>();
  previewImage: SafeUrl = '';

  constructor() {
    effect((onCleanup) => {
      const fg = this.data();
      if (!fg) return;

      const sub: Subscription = fg?.valueChanges
        ?.pipe(takeUntilDestroyed(this.destroyRef))
        .subscribe(() => {
          this.cdr.markForCheck();
        });

      onCleanup(() => sub && sub.unsubscribe());
    });
  }

  getStrByFormId(i: number): FormGroup<DocumentsFieldForm> {
    return this.data().controls.fields.at(i);
  }
}
