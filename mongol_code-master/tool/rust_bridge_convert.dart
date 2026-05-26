import 'dart:convert';
import 'dart:io';

import 'package:mongol_code/mongol_code.dart';

Future<void> main(List<String> args) async {
  if (args.length != 1) {
    stderr.writeln('Usage: rust_bridge_convert.dart <unicode-to-menksoft|menksoft-to-unicode>');
    exitCode = 64;
    return;
  }

  final input = await stdin.transform(utf8.decoder).join();
  switch (args.single) {
    case 'unicode-to-menksoft':
      stdout.write(convertUnicodeToMenksoft(input));
    case 'menksoft-to-unicode':
      stdout.write(convertMenksoftToUnicode(input));
    default:
      stderr.writeln('Unknown conversion mode: ${args.single}');
      exitCode = 64;
  }
}
