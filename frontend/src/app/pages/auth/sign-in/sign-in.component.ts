import { Component, DestroyRef, inject } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { AuthService } from '../../../core/auth/auth.service';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { LoadingBarService } from '@core/loading-bar/loading-bar.service';
import { InformerService } from '@core/services/informer.service';
import { PathWithSlash, RoutesPath } from 'src/app/app.routes';


@Component({
  standalone: true,
  selector: 'app-sign-in',
  imports: [
    ReactiveFormsModule,
    MatFormFieldModule, MatInputModule, MatButtonModule, MatCardModule, MatIconModule, RouterModule
  ],
  templateUrl: './sign-in.component.html',
  styleUrls: ['../auth.scss'],
})
export class SignInComponent {
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  private readonly informerSrv = inject(InformerService);
  private readonly loadingBarSrv = inject(LoadingBarService);

  readonly isLoading = toSignal(this.loadingBarSrv.show$);
  readonly form = new FormGroup({
    login: new FormControl<string>('', { nonNullable: true, validators: [Validators.required] }),
    password: new FormControl<string>('', { nonNullable: true, validators: [Validators.required] }),
  });
  get signUpLink(): string {
    return PathWithSlash(RoutesPath.SignUp);
  }

  onSubmit(): void {
    if (this.form.invalid) return;

    this.auth.signIn(this.form.getRawValue()).pipe(
      this.loadingBarSrv.withLoading(),
      takeUntilDestroyed(this.destroyRef)).subscribe({
        next: (res) => {
          if (res?.token) {
            this.auth.saveToken(res.token);
            this.router.navigateByUrl(PathWithSlash(RoutesPath.Dashboard));
          }
        },
        error: (err) => this.informerSrv.error(err?.error?.message, 'Ошибка авторизации'),
      });

  }
}
