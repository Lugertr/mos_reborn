import { ChangeDetectionStrategy, Component, inject, OnInit, signal } from '@angular/core';
import { Router, RouterOutlet } from '@angular/router';
import { NavbarComponent } from './components/navbar/navbar.component';
import { FooterComponent } from './components/footer/footer.component';
import { LoadingBarComponent } from '@core/loading-bar/loading-bar.component';
import { PathWithSlash, RoutesPath } from './app.routes';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
  imports: [
    RouterOutlet,
    NavbarComponent,
    FooterComponent,
    LoadingBarComponent
  ],
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush

})
export class AppComponent {
  private readonly router = inject(Router);
  isAboutPage = signal<boolean>(this.router.url === PathWithSlash(RoutesPath.About));
}
