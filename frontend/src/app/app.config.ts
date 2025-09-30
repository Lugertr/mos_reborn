import { ApplicationConfig, importProvidersFrom, inject, provideAppInitializer, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { routes } from './app.routes';
import { provideHttpClient, withFetch, withInterceptors, withInterceptorsFromDi } from '@angular/common/http';
import { PlatformLocation } from '@angular/common';
import { authInterceptor } from '@core/auth/auth-http-interceptor';
import { ToastrModule } from 'ngx-toastr';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { AuthService } from '@core/auth/auth.service';

export const getBaseHref: (plSrv: PlatformLocation) => string = (plSrv: PlatformLocation) => plSrv.getBaseHrefFromDOM();

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    provideAnimationsAsync(),
    importProvidersFrom(ToastrModule.forRoot()),
    provideHttpClient(withInterceptors([authInterceptor]), withInterceptorsFromDi(), withFetch()),
    provideAppInitializer(() => {
      const auth = inject(AuthService);
      return auth.updateToken();
    }),
  ]
};
