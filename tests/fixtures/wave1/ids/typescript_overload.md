# TypeScript Overloads

## Function f

```typescript
function f(x: string): string;
function f(x: number): number;
function f(x: string | number): string | number {
    return x;
}
```

TypeScript overload signatures — each overload variant gets a ``:f@<sig-hash>``
suffix to distinguish them.
