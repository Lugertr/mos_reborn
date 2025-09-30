import { Pipe, PipeTransform } from '@angular/core';
import { PrivacyType } from '../services/dashboard.service';

@Pipe({
  name: 'privacyLabel',
  standalone: true,
})
export class PrivacyLabelPipe implements PipeTransform {
  private readonly map: Record<string, string> = {
    [PrivacyType.Public]: 'Публичный',
    [PrivacyType.Private]: 'Приватный',
  };

  transform(value: PrivacyType | string | null | undefined): string {
    if (value == null || value === '') return '-';
    const key = String(value);
    return this.map[key] ?? key;
  }
}
