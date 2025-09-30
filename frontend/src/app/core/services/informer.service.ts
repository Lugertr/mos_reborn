import { Injectable } from '@angular/core';
import { IndividualConfig, ToastrService } from 'ngx-toastr';
import { HttpErrorResponse } from '@angular/common/http';
import { ServerItemError } from '@core/rest-http-client/rest-error-handlers';

@Injectable({ providedIn: 'root' })
export class InformerService {
  constructor(private toastr: ToastrService) {}

  error<T>(err: ServerItemError | HttpErrorResponse | string, title?: string, result?: T, override?: Partial<IndividualConfig>): T | null {
    let errMsg = '';

    switch (true) {
      case !err:
        break;
      case typeof err === 'string':
        errMsg = err;
        break;
      default:
        const errHandler = err?.error || err;
        errMsg = errHandler.message || errHandler.details || '';
        break;
    }

    this.toastr.error(errMsg, title, { ...override });
    return result || null;
  }

  warn(msg: string, title?: string, override?: Partial<IndividualConfig>): void {
    this.toastr.warning(msg, title, { ...override });
  }

  info(msg: string, title?: string, override?: Partial<IndividualConfig>): void {
    this.toastr.info(msg, title, { ...override });
  }

  success(msg: string, title?: string, override?: Partial<IndividualConfig>): void {
    this.toastr.success(msg, title, { ...override });
  }

  clearToasts(): void {
    this.toastr.clear();
  }
}
