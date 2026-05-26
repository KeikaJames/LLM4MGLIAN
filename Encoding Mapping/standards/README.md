# Standards Reference

This directory stores local standards material used while porting the encoding
mapping rules.

## GB/T 25914-2023

- Official page:
  `https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno=BD6429DE5A7FC782FAAE13938A07166E`
- Title: `信息技术 传统蒙古文名义字符、变形显现字符和控制字符使用规则`
- English title: `Information technology--Traditional Mongolian nominal characters, presentation characters and use rules of controlling characters`
- Status: current
- Release date: 2023-11-27
- Implementation date: 2024-06-01

The public standards system exposes online preview and download actions for this
standard, but the download flow requires CAPTCHA verification. Large standards
PDFs are intentionally ignored by git; use the official page as the source of
record.

## Onon Encoding References

- Onon input-method help documents three modes: national standard code (`MN`),
  Menksoft code (`MK`), and 民族事务委员会共享工程标准编码 (`MW`):
  `https://ime.onon.cn/help-index.html`
- Onon Windows changelog records MW support and interconversion with standard
  2010 and Menksoft codes:
  `https://ime.onon.cn/zh-CN/changelog/win`
- Onon web converter exposes GB2010, MW, MKL, and DLH modes:
  `https://mt.onon.cn/codeconvert`

Implementation note: Onon's current MW conversion output is Unicode codepoints
with required variation selectors for the newer standard form, not Menksoft
PUA. For tokenizer training we normalize it with
`normalize_to_nominal_unicode`, which removes glyph-only FVS selectors after
conversion while preserving MVS suffix separators.
