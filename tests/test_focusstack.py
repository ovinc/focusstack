"""Test focusstack module with pytest"""

from pathlib import Path
import focusstack


datafolder = Path(focusstack.__file__).parent / '..' / 'data/'
imgfolder = datafolder / 'input'


def test_focusstack():
    """Test that focus stacking outputs a merged image."""
    focusstack.stack(path=imgfolder, savepath=datafolder, use_sift=False)
    merged_file = datafolder / "merged.png"
    assert merged_file.exists()
