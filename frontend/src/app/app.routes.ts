import { Routes } from '@angular/router';

export enum RoutesPath {
  About = '',
  Conversion = 'conversion',
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
  { path: RoutesPath.Conversion, title: 'Конверция документов', icon: 'article_shortcut' },

];

export const routes: Routes = [
  {
    path: RoutesPath.About,
    loadComponent: () => import('./pages/about/about.component').then(c => c.AboutComponent),
    pathMatch: 'full',
  },
  {
    path: RoutesPath.Conversion,
    loadComponent: () => import('./pages/conversion/conversion.component').then(c => c.ConversionComponent),
  },
  { path: '**', redirectTo: RoutesPath.About },
];
