import { ChangeDetectionStrategy, Component } from '@angular/core';
import { DocumentComponent } from 'src/app/components/document/document.component';

@Component({
  selector: 'app-conversion',
  templateUrl: './conversion.component.html',
  styleUrl: './conversion.component.scss',
  standalone: true,
  imports: [DocumentComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ConversionComponent {

}
