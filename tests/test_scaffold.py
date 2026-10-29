def test_package_imports():
    import nuzline

    assert nuzline.__version__ == "0.0.1"


def test_jax_is_cpu_and_32bit():
    import jax
    import jax.numpy as jnp

    assert jax.default_backend() == "cpu"
    assert jnp.arange(3).dtype == jnp.int32
