"""Test script to verify ALCF unified client."""

from research_agent.alcf_client import get_alcf_client
from research_agent.auth import check_token_validity


def test_authentication():
    """Test Globus authentication."""
    print("Testing Globus authentication...")

    if check_token_validity():
        print("✓ Authentication token is valid")
        return True
    else:
        print("✗ Authentication failed")
        print("Run: python inference_auth_token.py authenticate")
        return False


def test_chat_completion():
    """Test LLM chat completion."""
    print("\nTesting LLM chat completion...")

    try:
        client = get_alcf_client()

        response = client.chat(
            messages=[{"role": "user", "content": "What is the capital of France? Answer in one word."}],
            max_tokens=10
        )

        print(f"✓ Chat completion successful")
        print(f"  Response: {response}")
        return True
    except Exception as e:
        print(f"✗ Chat completion failed: {e}")
        return False


def test_embeddings():
    """Test embedding generation (HuggingFace by default)."""
    print("\nTesting embedding generation (HuggingFace)...")

    try:
        client = get_alcf_client()

        # Single embedding
        embedding = client.embed("gene regulation in E. coli")
        print(f"✓ Single embedding successful (dim={len(embedding)})")
        print(f"  Provider: {client.embedding_provider}")

        # Batch embeddings
        embeddings = client.embed_batch([
            "lexA represses recA",
            "crp activates lacZ"
        ])
        print(f"✓ Batch embeddings successful (count={len(embeddings)})")

        return True
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_client():
    """Test that single client can do both operations."""
    print("\nTesting unified client (both LLM and embeddings)...")

    try:
        # Single client instance
        client = get_alcf_client()

        # Use for chat
        chat_response = client.chat(
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            max_tokens=5
        )

        # Use for embedding
        embedding = client.embed("test text")

        print(f"✓ Unified client works for both operations")
        print(f"  Chat response: {chat_response}")
        print(f"  Embedding dim: {len(embedding)}")

        return True
    except Exception as e:
        print(f"✗ Unified client test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("ALCF Unified Client Test Suite")
    print("="*60)

    results = []

    # Test authentication first
    if not test_authentication():
        print("\n⚠ Authentication required before running other tests")
        return

    # Run tests
    results.append(("Chat Completion", test_chat_completion()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Unified Client", test_unified_client()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
