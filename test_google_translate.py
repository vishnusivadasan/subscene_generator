#!/usr/bin/env python3
"""
Test script to validate Google Translate approach for subtitle translation.
Tests single-line and bundled translation strategies with async googletrans.
"""

import asyncio
from googletrans import Translator
import time

# Sample Japanese lines from the SRT file
test_lines = [
    "なにやってんのしよう",
    "じっとしてないとだめじゃない",
    "だって大会近いし",
    "そんなこと言って",
    "またこの膝が赤したらどうすんの",
]

async def test_line_by_line():
    """Test line-by-line translation (5 API calls)."""
    translator = Translator()
    print("\n[Test 1] Line-by-line translation (5 API calls):")
    print("-" * 80)

    start_time = time.time()
    results = []

    for i, line in enumerate(test_lines, 1):
        try:
            result = await translator.translate(line, src='ja', dest='en')
            results.append(result.text)
            print(f"{i}. JA: {line}")
            print(f"   EN: {result.text}\n")
            await asyncio.sleep(0.2)  # Small delay to avoid rate limits
        except Exception as e:
            print(f"{i}. ERROR: {e}\n")
            results.append(f"[ERROR: {line}]")

    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")
    return results, elapsed

async def test_bundled():
    """Test bundled translation with newline separator (1 API call)."""
    translator = Translator()
    print("\n" + "=" * 80)
    print("[Test 2] Bundled translation with newline separator (1 API call):")
    print("-" * 80)

    start_time = time.time()
    bundled_text = "\n".join(test_lines)
    print(f"Bundled input:\n{bundled_text}\n")

    results = []
    try:
        result = await translator.translate(bundled_text, src='ja', dest='en')
        bundled_output = result.text
        results = bundled_output.split('\n')

        print(f"Bundled output:\n{bundled_output}\n")
        print("Parsed results:")
        for i, (ja, en) in enumerate(zip(test_lines, results), 1):
            print(f"{i}. JA: {ja}")
            print(f"   EN: {en}\n")
    except Exception as e:
        print(f"ERROR: {e}")

    elapsed = time.time() - start_time
    print(f"Time taken: {elapsed:.2f}s")
    return results, elapsed

async def main():
    print("=" * 80)
    print("GOOGLE TRANSLATE TEST (Async Version)")
    print("=" * 80)

    # Run tests
    line_by_line_results, line_by_line_time = await test_line_by_line()
    bundled_results, bundled_time = await test_bundled()

    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("-" * 80)
    print(f"Line-by-line: {line_by_line_time:.2f}s ({len(test_lines)} API calls)")
    print(f"Bundled:      {bundled_time:.2f}s (1 API call)")
    if bundled_time > 0:
        print(f"Speedup:      {line_by_line_time / bundled_time:.2f}x faster")
    print(f"\nAPI calls saved: {len(test_lines) - 1} ({(len(test_lines) - 1) / len(test_lines) * 100:.0f}% reduction)")

    # Quality check
    print("\n" + "=" * 80)
    print("QUALITY CHECK:")
    print("-" * 80)
    if len(bundled_results) == len(test_lines):
        print("✅ Correct number of lines preserved")
        all_match = all(a == b for a, b in zip(line_by_line_results, bundled_results))
        if all_match:
            print("✅ Translations identical (bundling has no quality loss)")
        else:
            print("⚠️  Some translations differ:")
            for i, (single, bundle) in enumerate(zip(line_by_line_results, bundled_results), 1):
                if single != bundle:
                    print(f"  Line {i}:")
                    print(f"    Single: {single}")
                    print(f"    Bundle: {bundle}")
    else:
        print(f"❌ Line count mismatch! Expected {len(test_lines)}, got {len(bundled_results)}")
        print("   Bundling may not be reliable for this use case")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("-" * 80)
    if len(bundled_results) == len(test_lines):
        print("✅ Bundling strategy is VIABLE")
        print("   - Preserves line count")
        print("   - Significantly faster")
        print("   - Reduces API calls by 80%")
        print("\n💡 RECOMMENDATION: Use bundling strategy (5-10 lines per batch)")
    else:
        print("⚠️  Bundling strategy needs refinement")
        print("   - Consider using delimiters like ' ||| ' instead of newlines")
        print("   - Or fall back to line-by-line translation")

if __name__ == "__main__":
    asyncio.run(main())
