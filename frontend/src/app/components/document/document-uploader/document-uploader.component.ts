import {
  Component,
  ChangeDetectionStrategy,
  output,
  inject,
  HostBinding,
} from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { FileConversionService, FileData } from '../services/file-conversion.service';

@Component({
  selector: 'app-document-uploader',
  standalone: true,
  imports: [MatButtonModule, MatIconModule],
  templateUrl: './document-uploader.component.html',
  styleUrls: ['./document-uploader.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class DocumentUploaderComponent {
  @HostBinding('class.mat-card') cssClass = true;

  readonly filesSelected = output<FileData[]>();
  private readonly fileConversion = inject(FileConversionService);

  async onSelectFiles(ev: Event): Promise<void> {
    const files = await this.fileConversion.onSelectFiles(ev);
    this.filesSelected.emit(files);
  }
}
