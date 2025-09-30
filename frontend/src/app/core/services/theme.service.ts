// theme.service.ts
import { Inject, Injectable, effect } from '@angular/core';
import { DOCUMENT, isPlatformBrowser } from '@angular/common';
import { Renderer2, RendererFactory2 } from '@angular/core';
import { signal, Signal } from '@angular/core';

const STORAGE_KEY = 'theme';
const DARK_CLASS = 'dark';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  private isDarkSignal = signal(false);
  readonly isDark: Signal<boolean> = this.isDarkSignal.asReadonly();

  private renderer: Renderer2;
  private storageAvailable: boolean;

  constructor(
    @Inject(DOCUMENT) private document: Document,
    rendererFactory: RendererFactory2
  ) {
    this.renderer = rendererFactory.createRenderer(null, null);
    this.storageAvailable = typeof window?.localStorage !== 'undefined';
    this.initialize();
  }

  toggle(): void {
    const next = !this.isDarkSignal();
    this.isDarkSignal.set(next);
    if (this.storageAvailable) {
      try {
        localStorage.setItem(STORAGE_KEY, next ? 'dark' : 'light');
      } catch { }
    }
  }

  private initialize(): void {
    if (this.storageAvailable) {
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        this.isDarkSignal.set(saved === 'dark');
      } catch {
        this.isDarkSignal.set(false);
      }
    } else {
      if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
        this.isDarkSignal.set(true);
      }
    }

    effect(() => {
      if (this.isDarkSignal()) {
        this.renderer.addClass(this.document.body, DARK_CLASS);
      } else {
        this.renderer.removeClass(this.document.body, DARK_CLASS);
      }
    });
  }
}
