/**
 * Ротрисовывает выбранный документ (изображение) и даёт доступ к его полям/координатам.
 */
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
import { CdkDrag, DragDropModule } from '@angular/cdk/drag-drop';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatMenuModule } from '@angular/material/menu';

/**
 * Выводит изображения и поля для редактирования
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
    CdkDrag
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
  // В компонент прилетает FormGroup документа; OnPush + Signals = предсказуемые апдейты
  readonly data = input<FormGroup<DocumentForm>>();

  /**
   * Предпросмотр через ObjectURL.
   * Важно: computed пересчитывается при изменении input-сигнала; ниже добавлен effect для revoke().
   */
  previewImg: Signal<SafeUrl | null> = computed(() => {
    const imgFile = this.data()?.value?.file;
    if (imgFile) {
      const objUrl = URL.createObjectURL(imgFile);
      return this.sanitizer.bypassSecurityTrustUrl(objUrl);
    }
    return null;
  });

  // Вспомогательный стрим для ручного рефреша (оставлен на будущее, см. шаблон)
  reload$ = new Subject<void>();
  // legacy: не используется напрямую, оставлено для совместимости с шаблоном
  previewImage: SafeUrl = '' as any;

  constructor() {
    // Авто-обновление ChangeDetector (OnPush) при изменении формы
    effect((onCleanup) => {
      const fg = this.data();
      if (!fg) return;

      const sub: Subscription = fg.valueChanges
        ?.pipe(takeUntilDestroyed(this.destroyRef))
        .subscribe(() => this.cdr.markForCheck());

      onCleanup(() => sub && sub.unsubscribe());
    });

    // Обязательно освобождаем ObjectURL при смене файла/уничтожении компонента
    effect((onCleanup) => {
      const file = this.data()?.value?.file;
      if (!file) return;
      const url = URL.createObjectURL(file);
      onCleanup(() => URL.revokeObjectURL(url));
    });
  }

  // Достаём form группы строк по индексу
  getStrByFormId(i: number): FormGroup<DocumentsFieldForm> {
    return this.data().controls.fields.at(i);
  }
}
