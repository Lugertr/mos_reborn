import { MatTableDataSource } from '@angular/material/table';
import { isFilledArray, uniqWith } from '@core/helpers/array-utils';
import { Observable, map } from 'rxjs';

export type FilterObj<T> = { [key in keyof T]?: unknown };

export class TableDataSource<T> extends MatTableDataSource<T> {
  private _filterObj: FilterObj<T>;

  get filterObj(): FilterObj<T> {
    return this._filterObj;
  }
  set filterObj(f: FilterObj<T>) {
    this._filterObj = f;
    const filterVal = this.filter;
    this.filter = filterVal;
  }

  override _filterData(data: T[]): T[] {
    if (this.filterObj) {
      this.filteredData = data.filter(obj => this.filterObjPredicate(obj, this.filterObj));
    } else {
      this.filteredData = !this.filter ? data : data.filter(obj => this.filterPredicate(obj, this.filter));
    }

    if (this.paginator) {
      this._updatePaginator(this.filteredData.length);
    }

    return this.filteredData;
  }

  filterObjPredicate: (data: T, filter: FilterObj<T>) => boolean = (data: T, filter: FilterObj<T>): boolean => {
    for (const key of Object.keys(filter)) {
      const [v, f] = [data[key], filter[key]];
      if (isFilledArray(f)) {
        if (f.indexOf(v) === -1) {
          return false;
        }
      } else {
        const valStr = v ? `${v}`.toLowerCase() : '';
        const filterStr = f ? `${f}`.trim().toLowerCase() : '';
        if (valStr.indexOf(filterStr) === -1) {
          return false;
        }
      }
    }
    return true;
  };

  getSetOf<K>(convert: (item: T) => K, equalBy?: (a: K, b: K) => boolean): Observable<K[]> {
    const data$ = this.connect().pipe(map(() => this.data));
    return data$.pipe(
      map(items => {
        if (equalBy === undefined) {
          const s = new Set<K>();
          for (const item of items) {
            s.add(convert(item));
          }
          return Array.from(s);
        } else {
          return uniqWith(items.map(convert), equalBy);
        }
      }),
    );
  }

  push(...rows: T[]): void {
    this.data = [...this.data, ...rows];
  }

  unshift(...rows: T[]): void {
    this.data.unshift(...rows);
    this.data = this.data.slice();
  }

  remove(row: T): void {
    const index = this.data.indexOf(row);
    if (index !== -1) {
      this.data.splice(index, 1);
      this.data = this.data.slice();
    }
  }

  reset(rows: T[] = []): void {
    this.data = rows || [];
  }

  count(): number {
    return this.data.length;
  }
}
