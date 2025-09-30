import { HttpErrorResponse, HttpStatusCode } from '@angular/common/http';
import { isFilledArray } from '@core/helpers/array-utils';
import { Observable, fromEvent, switchMap, throwError } from 'rxjs';

export interface Data<D = unknown> {
  responseStatusCode: HttpStatusCode;
  data?: D;
  errors?: Error[];
}

export interface Error {
  error?: string;
  message?: string;
  detail?: string;
}

export class RestHttpResponse<D = unknown> implements Data {
  responseStatusCode: HttpStatusCode;
  data?: D;
  errors?: ServerItemError[];

  constructor(data: Data) {
    Object.assign(this, data);
  }

  get hasError(): boolean {
    return this.responseStatusCode !== HttpStatusCode.Ok && this.responseStatusCode !== HttpStatusCode.NoContent;
  }

  toString(): string {
    let result = '';
    if (this.hasError && isFilledArray(this.errors)) {
      result = JSON.stringify(this.errors, null, 2);
    } else if (this.data) {
      result = JSON.stringify(this.data, null, 2);
    }
    return result;
  }
}

export interface ServerItemError<D = unknown> {
  error: string;
  message?: string;
  detail?: string;
  status?: HttpStatusCode;
  data?: D;
}

export type TitledServerError = ServerItemError & { title?: string };

export interface ServerErrors {
  errors: ServerItemError[];
}

/**
 * Класс для удобства приведения ошибки в строку.
 *
 * @example catchError((err: ServerItemError) { this.informer.error(err.toString()) }
 * @example catchError((err: ServerItemError) { this.informer.error(`${err}`) }
 */
export class RestError<D = unknown> implements ServerItemError<D> {
  error: string;
  message: string;
  detail: string;
  status: number;
  data: D;

  constructor(err: ServerItemError<D>) {
    Object.assign(this, err);
  }

  toString(): string {
    let msgErr: string;
    let detail: string = this.detail;
    if (this.status === 0) {
      msgErr = 'Network error';
      detail = this.message;
    } else {
      msgErr = `Err: ${this.message || this.error}`;
    }
    const err = [msgErr];
    if (detail) {
      err.push(`detail: ${detail}`);
    }
    return err.join(', ');
  }
}

const HTTP_NETWORK_ERROR_STATUS_CODE = 0;

export function handleRestHttpError(resp: HttpErrorResponse): Observable<never> {
  switch (resp.status) {
    case HttpStatusCode.NotFound:
      return throwError(
        () =>
          new RestError({
            error: resp.statusText,
            detail: resp.message,
            status: resp.status,
          }),
      );
    case HTTP_NETWORK_ERROR_STATUS_CODE:
      return throwError(
        () =>
          new RestError({
            error: 'network.error',
            message: 'Ошибка сети, проверьте подключение.',
            detail: resp.message,
            status: resp.status,
          }),
      );
  }
  const body = resp.error;

  if (body && Array.isArray(body?.errors) && body.errors.length > 0) {
    const itemError: ServerItemError = body.errors[0];

    if (body?.data) {
      itemError.data = body.data;
    }

    return throwError(() => new RestError({ ...itemError, status: resp.status }));
  } else if (body && typeof body?.error === 'string') {
    return throwError(
      () =>
        new RestError({
          error: body.error,
          detail: resp.message,
          status: resp.status,
        }),
    );
  }

  return throwError(
    () =>
      new RestError({
        error: body,
        detail: resp.message,
        status: resp.status,
      }),
  );
}

export function handleRestHttpMultipleErrors(resp: HttpErrorResponse): Observable<never> {
  const restError = new RestHttpResponse({
    responseStatusCode: resp.status,
  });
  const body = resp.error;

  if (resp.status !== HttpStatusCode.NotFound && isFilledArray(body?.errors)) {
    if (body?.data) {
      restError.data = body.data;
    }
    restError.errors = body.errors;
  } else {
    restError.errors = [{ error: body, message: resp.message }];
  }

  return throwError(() => restError);
}

// Temporary solution, https://github.com/angular/angular/issues/19148
export function handleRestHttpBlobError(error: HttpErrorResponse): Observable<never> {
  const cType = error.headers.get('Content-Type');
  if (typeof cType === 'string' && cType.split(/\s*;\s*/).indexOf('application/json') !== -1) {
    const reader = new FileReader();
    reader.readAsText(error.error);
    return fromEvent(reader, 'loadend').pipe(
      switchMap(ev =>
        handleRestHttpError({
          ...error,
          error: JSON.parse(ev.target['result']),
        }),
      ),
    );
  } else {
    return handleRestHttpError(error);
  }
}
