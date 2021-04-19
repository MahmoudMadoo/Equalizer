![](screenshoot/equilizer.gif)

### There is Modification in pyqtgraph package 
module plotItem.py lines 1175 and 1118 
```python
 if self._exportOpts is False and self.mouseHovering and not self.buttonsHidden and not all(self.vb.autoRangeEnabled()):
                if self.autoBtn is not None:
                    self.autoBtn.show()
            else:
                if self.autoBtn is not None:
                    self.autoBtn.hide()
```
