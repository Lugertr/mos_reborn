import { HttpClient, HttpContext, HttpHeaders, HttpParams, HttpResponse, HttpStatusCode } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, catchError, map } from 'rxjs';
import { handleRestHttpError, handleRestHttpMultipleErrors } from './rest-error-handlers';

interface ServerBody<T> {
  data?: T;
}

@Injectable({ providedIn: 'root' })
export class RestHttpClient {
  constructor(private http: HttpClient) {}

  get<T>(
    url: string,
    options?: {
      headers?: HttpHeaders | Record<string, string | string[]>;
      context?: HttpContext;
      params?: HttpParams | Record<string, string | string[]>;
      catchMultipleErrors?: boolean;
    },
  ): Observable<T> {
    return this.http.get<ServerBody<T>>(url, { ...options, observe: 'response' }).pipe(this.processResponse(options?.catchMultipleErrors));
  }

  post<T>(
    url: string,
    body: unknown | null,
    options?: {
      headers?: HttpHeaders | Record<string, string | string[]>;
      context?: HttpContext;
      params?: HttpParams | Record<string, string | string[]>;
      catchMultipleErrors?: boolean;
    },
  ): Observable<T> {
    return this.http
      .post<ServerBody<T>>(url, body, { ...options, observe: 'response' })
      .pipe(this.processResponse(options?.catchMultipleErrors));
  }

  patch<T>(
    url: string,
    body: unknown | null,
    options?: {
      headers?: HttpHeaders | Record<string, string | string[]>;
      context?: HttpContext;
      params?: HttpParams | Record<string, string | string[]>;
      catchMultipleErrors?: boolean;
    },
  ): Observable<T> {
    return this.http
      .patch<ServerBody<T>>(url, body, { ...options, observe: 'response' })
      .pipe(this.processResponse(options?.catchMultipleErrors));
  }

  private processResponse<T>(multiple?: boolean): (resp: Observable<HttpResponse<ServerBody<T>>>) => Observable<T> {
    return resp =>
      resp.pipe(
        map(res => {
          if (res.status === HttpStatusCode.NoContent) {
            return null;
          }

          if (res.body instanceof Object && Object.prototype.hasOwnProperty.call(res.body, 'data')) {
            return res.body.data;
          }

          return (res.body as T) || null;
        }),
        catchError(multiple ? handleRestHttpMultipleErrors : handleRestHttpError),
      );
  }
}
