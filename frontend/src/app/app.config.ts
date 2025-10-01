import { ApplicationConfig, importProvidersFrom, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { routes } from './app.routes';
import { provideHttpClient, withFetch, withInterceptorsFromDi } from '@angular/common/http';
import { PlatformLocation } from '@angular/common';
import { ToastrModule } from 'ngx-toastr';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

export const getBaseHref: (plSrv: PlatformLocation) => string = (plSrv: PlatformLocation) => plSrv.getBaseHrefFromDOM();

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    provideAnimationsAsync(),
    importProvidersFrom(ToastrModule.forRoot()),
    provideHttpClient(withInterceptorsFromDi(), withFetch()),
  ]
};
