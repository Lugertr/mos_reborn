import { Component, DestroyRef, inject } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterModule } from '@angular/router';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { AuthService } from '../../../core/auth/auth.service';
import { LoadingBarService } from '@core/loading-bar/loading-bar.service';
import { InformerService } from '@core/services/informer.service';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { PathWithSlash, RoutesPath } from 'src/app/app.routes';
import { catchError, mergeMap, of } from 'rxjs';

@Component({
  standalone: true,
  selector: 'app-sign-up',
  imports: [
    ReactiveFormsModule,
    MatFormFieldModule, MatInputModule, MatButtonModule, MatCardModule, MatSnackBarModule, RouterModule
  ],
  templateUrl: './sign-up.component.html',
  styleUrls: ['../auth.scss'],
})
export class SignUpComponent {
  private readonly auth = inject(AuthService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  private readonly informerSrv = inject(InformerService);
  private readonly loadingBarSrv = inject(LoadingBarService);

  readonly isLoading = toSignal(this.loadingBarSrv.show$);
  form = new FormGroup({
    login: new FormControl('', [Validators.required]),
    password: new FormControl('', [Validators.required, Validators.minLength(6)]),
    full_name: new FormControl('', [Validators.required]),
  });
  get signInLink(): string {
    return PathWithSlash(RoutesPath.SignIn);
  }

  onSubmit(): void {
    if (this.form.invalid) return;

    const payload = {
      login: this.form.value.login!,
      password: this.form.value.password!,
      full_name: this.form.value.full_name,
    };

    this.auth.signUp(payload).pipe(
      mergeMap(() => this.auth.signIn(payload).pipe(catchError((err) => {
        this.informerSrv.error(err?.error?.message, 'Ошибка регистрации');
        return of(null);
      }))),
      this.loadingBarSrv.withLoading(),
      takeUntilDestroyed(this.destroyRef)).subscribe({
        next: (res) => {
          if (res?.token) {
            this.auth.saveToken(res.token);
            this.router.navigateByUrl(PathWithSlash(RoutesPath.Dashboard));
          }
        },
        error: (err) => this.informerSrv.error(err?.error?.message, 'Ошибка регистрации'),
      });
  }
}
