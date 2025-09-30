import { BreakpointObserver, Breakpoints } from '@angular/cdk/layout';
import { ChangeDetectionStrategy, Component, DestroyRef, EventEmitter, inject, Output, PLATFORM_ID, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatToolbarModule } from '@angular/material/toolbar';
import { Router, RouterModule } from '@angular/router';
import { AuthService } from '@core/auth/auth.service';
import { ThemeService } from '@core/services/theme.service';
import { MENU_ITEMS, MenuItem, PathWithSlash, RoutesPath } from 'src/app/app.routes';

@Component({
  selector: 'app-navbar',
  imports: [
    MatListModule,
    MatToolbarModule,
    MatButtonModule,
    MatIconModule,
    RouterModule,
],
  templateUrl: './navbar.component.html',
  styleUrl: './navbar.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class NavbarComponent {
  @Output() menuToggle = new EventEmitter<void>();
  readonly themeService = inject(ThemeService);
  private readonly router = inject(Router);
  private readonly auth = inject(AuthService);
  private readonly bp = inject(BreakpointObserver);
  private readonly destroy$ = inject(DestroyRef);

  readonly isMobile = signal(false);
  menuItems: MenuItem[] = MENU_ITEMS;

  ngOnInit(): void {
    this.isMobile.set(this.bp.isMatched(Breakpoints.Handset));

    this.bp.observe([Breakpoints.Handset]).pipe(takeUntilDestroyed(this.destroy$)).subscribe((result) => {
      this.isMobile.set(result.matches);
    });
  }

  get userLink(): string {
    return PathWithSlash(RoutesPath.About);
  }

  get isAuth(): boolean {
    return !!this.auth.token();
  }

  signOut(): void {
    this.auth.clearToken();
    this.router.navigateByUrl(PathWithSlash(RoutesPath.SignIn));
  }

  menuToggleClick(): void {
    this.menuToggle.emit();
  }
}
