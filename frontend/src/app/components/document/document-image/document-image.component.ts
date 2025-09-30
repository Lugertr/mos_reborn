import { CommonModule } from '@angular/common';
import { Component, ChangeDetectionStrategy, Input, input } from '@angular/core';
import { FormArray, FormControl, FormGroup, ReactiveFormsModule } from '@angular/forms';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { DocumentsField } from '../document.component';

/**
 * Displays the image and overlays form inputs at coordinate positions.
 * The coordinates are interpreted as CSS pixels relative to the natural image size.
 */
@Component({
  selector: 'app-document-image',
  standalone: true,
  imports: [ReactiveFormsModule, MatFormFieldModule, MatInputModule],
  templateUrl: './document-image.component.html',
  styleUrls: ['./document-image.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DocumentImageComponent {
  imageSrc = input.required<string>();
  fieldsFormArray = input.required<FormArray<FormGroup<DocumentsField>>>();

  getStrByFormId(i: number): FormGroup<DocumentsField> {
    return this.fieldsFormArray().at(i);
  }
}
