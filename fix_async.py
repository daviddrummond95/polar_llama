#!/usr/bin/env python3
"""Fix async blocks in expressions.rs"""
import re

with open('src/expressions.rs', 'r') as f:
    content = f.read()

# Pattern: run_async(async { fetch_XXX(...) })
# Need to add .await before the last )

# Find all run_async(async { ... }) patterns and add .await
def fix_async_block(match):
    full = match.group(0)
    # Count parentheses to find the matching close
    open_count = 0
    start_func = full.find('fetch')

    for i in range(start_func, len(full)):
        if full[i] == '(':
            open_count += 1
        elif full[i] == ')':
            open_count -= 1
            if open_count == 0:
                # This is the closing paren of the function call
                # Insert .await before it
                return full[:i] + '.await' + full[i:]

    return full

# Match run_async(async { fetch... up to the end
pattern = r'run_async\(async \{ (fetch_[a-z_]+|crate::utils::fetch_[a-z_]+)\([^)]*\)\)'

# For simple cases (no nested parens in args)
content = re.sub(
    r'run_async\(async \{ (fetch_[a-z_]+\([^)]+\))\)',
    r'run_async(async { \1.await })',
    content
)

# For complex cases with nested function calls
content = re.sub(
    r'run_async\(async \{ (fetch_[a-z_]+\([^)]+\([^)]+\)[^)]*\))\)',
    r'run_async(async { \1.await })',
    content
)

# For crate::utils:: cases
content = re.sub(
    r'run_async\(async \{ (crate::utils::fetch_[a-z_]+\([^)]+\))\)',
    r'run_async(async { \1.await })',
    content
)

content = re.sub(
    r'run_async\(async \{ (crate::utils::fetch_[a-z_]+\([^)]+\([^)]+\)[^)]*\))\)',
    r'run_async(async { \1.await })',
    content
)

with open('src/expressions.rs', 'w') as f:
    f.write(content)

print("Fixed async blocks")
