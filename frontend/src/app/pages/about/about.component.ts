import { NgStyle } from '@angular/common';
import { ChangeDetectionStrategy, Component } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { RouterModule } from '@angular/router';
import { PathWithSlash, RoutesPath } from 'src/app/app.routes';

@Component({
  selector: 'app-about',
  templateUrl: './about.component.html',
  styleUrl: './about.component.scss',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatIconModule, MatButtonModule,

    RouterModule,
  ]
})
export class AboutComponent {

  get dashboardUrl(): string {
    return PathWithSlash(RoutesPath.Dashboard);
  }

  get converterUrl(): string {
    return PathWithSlash(RoutesPath.Conversion);
  }
}
