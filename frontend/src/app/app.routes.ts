import { Routes } from '@angular/router';
import { authGuard } from '@core/auth/auth.guard';

export enum RoutesPath {
  About = '',
  Dashboard = 'dashboard',
  Conversion = 'conversion',
  SignIn = 'sign-in',
  SignUp = 'sign-up',
  Profile = 'profile',
}

export function PathWithSlash(route: RoutesPath): string {
  return '/' + route;
}

export interface MenuItem {
  path: RoutesPath;
  title: string;
  icon?: string;
  private?: boolean
}

export const MENU_ITEMS: MenuItem[] = [
  { path: RoutesPath.About, title: 'О проекте', icon: 'home' },
  { path: RoutesPath.Dashboard, title: 'Дашбоард', icon: 'dashboard', private: true },
  { path: RoutesPath.Conversion, title: 'Конверция документов', icon: 'article_shortcut' },

];

export const routes: Routes = [
  {
    path: RoutesPath.About,
    loadComponent: () => import('./pages/about/about.component').then(c => c.AboutComponent),
    pathMatch: 'full',
  },
  {
    path: RoutesPath.Dashboard,
    loadComponent: () => import('./pages/dashboard/dashboard.component').then(c => c.DashboardComponent),
    canActivate: [authGuard],
  },
  {
    path: RoutesPath.Conversion,
    loadComponent: () => import('./pages/conversion/conversion.component').then(c => c.ConversionComponent),
  },
  {
    path: RoutesPath.SignIn,
    loadComponent: () => import('./pages/auth/sign-in/sign-in.component').then(c => c.SignInComponent),
  },
  {
    path: RoutesPath.SignUp,
    loadComponent: () => import('./pages/auth/sign-up/sign-up.component').then(c => c.SignUpComponent),
  },
  { path: '**', redirectTo: RoutesPath.About },
];
