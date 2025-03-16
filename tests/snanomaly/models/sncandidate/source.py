import pytest
from hypothesis import given
from hypothesis import strategies as st

from snanomaly.models.sncandidate.source import Source


def test_source_init_with_required_params():
    """Test Source initialization with only required parameters."""
    source = Source(name="Test Source", alias=123)
    assert source.name == "Test Source"
    assert source.alias == 123
    assert source.url is None
    assert source.bibcode is None
    assert source.doi is None
    assert source.arxivid is None
    assert source.secondary is None
    assert source.acknowledgement is None


def test_source_init_with_all_params():
    """Test Source initialization with all parameters."""
    source = Source(
        name="Test Source",
        alias=123,
        url="https://example.com",
        bibcode="2023ApJ...123..456",
        doi="10.1234/5678",
        arxivid="2201.12345",
        secondary=False,
        acknowledgement="Thank the researchers",
    )

    assert source.name == "Test Source"
    assert source.alias == 123
    assert source.url == "https://example.com"
    assert source.bibcode == "2023ApJ...123..456"
    assert source.doi == "10.1234/5678"
    assert source.arxivid == "2201.12345"
    assert source.secondary is False
    assert source.acknowledgement == "Thank the researchers"


def test_source_equality():
    """Test equality comparison between Source objects."""
    source1 = Source(name="Same", alias=42, url="https://example.com")
    source2 = Source(name="Same", alias=42, url="https://example.com")
    source3 = Source(name="Different", alias=42)

    assert source1 == source2
    assert source1 != source3


def test_source_missing_required_params():
    """Test that Source raises when missing required parameters."""
    with pytest.raises(TypeError):
        Source()

    with pytest.raises(TypeError):
        Source(name="Missing alias")

    with pytest.raises(TypeError):
        Source(alias=42)


# Define hypothesis strategies for Source parameters
name_st = st.text(min_size=1)  # Non-empty strings for name
alias_st = st.integers()  # Any integer for alias
url_st = st.one_of(st.none(), st.text())
bibcode_st = st.one_of(st.none(), st.text())
doi_st = st.one_of(st.none(), st.text())
arxivid_st = st.one_of(st.none(), st.text())
secondary_st = st.one_of(st.none(), st.booleans())
acknowledgement_st = st.one_of(st.none(), st.text())


@given(
    name=name_st,
    alias=alias_st,
    url=url_st,
    bibcode=bibcode_st,
    doi=doi_st,
    arxivid=arxivid_st,
    secondary=secondary_st,
    acknowledgement=acknowledgement_st,
)
def test_source_property_based(name, alias, url, bibcode, doi, arxivid, secondary, acknowledgement):
    """Property-based testing for Source with various inputs."""
    source = Source(
        name=name,
        alias=alias,
        url=url,
        bibcode=bibcode,
        doi=doi,
        arxivid=arxivid,
        secondary=secondary,
        acknowledgement=acknowledgement,
    )

    assert source.name == name
    assert source.alias == alias
    assert source.url == url
    assert source.bibcode == bibcode
    assert source.doi == doi
    assert source.arxivid == arxivid
    assert source.secondary == secondary
    assert source.acknowledgement == acknowledgement


def test_source_repr():
    """Test string representation of Source object."""
    source = Source(name="Test", alias=987)
    repr_str = repr(source)

    assert "Test" in repr_str
    assert "987" in repr_str
    assert "Source" in repr_str
