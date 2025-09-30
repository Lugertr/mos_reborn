import { inject, Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { catchError, Observable, of, tap } from 'rxjs';
import { InformerService } from '@core/services/informer.service';

export interface SignUpInput {
  login: string;
  password: string;
  role_id?: number | null;
  full_name?: string | null;
}

export interface SignInInput {
  login: string;
  password: string;
}

export interface TokenResponse {
  token?: string;
}

export interface AuthResponse extends TokenResponse {
  user?: unknown;
}

const TOKEN_KEY = 'auth_token';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly http = inject(HttpClient);
  private readonly informerSrv = inject(InformerService);

  token = signal<string | null>(this.getToken());

  signUp(body: SignUpInput): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`/auth/sign-up`, body);
  }

  signIn(body: SignInInput): Observable<AuthResponse> {
    return this.http.post<AuthResponse>(`/auth/sign-in`, body);
  }

  updateToken(): Observable<TokenResponse> {
    const token = this.getToken();
    if (token) {
      this.token.set(token);
      return this.http.post<TokenResponse>(`/auth/refresh-token`, {}).pipe(catchError((err) => {
        this.informerSrv.error(err?.error?.message, 'Ошибка проверки токена');
        this.clearToken();
        return of(null);
      }),
        tap((newToken) => {
          if (newToken?.token) {
            this.saveToken(newToken.token);
          }
        })
      );
    } else {
      this.clearToken();
      return of(null);
    }
  }

  saveToken(token: string): void {
    localStorage.setItem(TOKEN_KEY, token);
    this.token.set(token);
  }

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  clearToken(): void {
    localStorage.removeItem(TOKEN_KEY);
    this.token.set(null);
  }

  isAuthenticated(): boolean {
    return !!this.getToken();
  }
}
