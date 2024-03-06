class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        for i in range(n, 1, -1):  # i is number of alphabets
            for j in range(0, n - i+1):  # j is the possibility
                s_dash = s[j: j + i]
                if len(s_dash) % 2 == 0:
                    m = s_dash[0: int(len(s_dash) / 2)]
                    l = ''.join(reversed(s_dash[int(len(s_dash) / 2):]))
                    if m == l:
                        return s_dash
                else:
                    m = s_dash[0: int(len(s_dash) / 2)]
                    l = ''.join(reversed(s_dash[int(len(s_dash) / 2+1):]))
                    if m == l:
                        return s_dash
        return s[0]


S = Solution()
print(S.longestPalindrome("gphyvqruxjmwhonjjrgumxjhfyupajxbjgthzdvrdqmdouuukeaxhjumkmmhdglqrrohydrmbvtuwstgkobyzjjtdtjroqpyusfsbjlusekghtfbdctvgmqzeybnwzlhdnhwzptgkzmujfldoiejmvxnorvbiubfflygrkedyirienybosqzrkbpcfidvkkafftgzwrcitqizelhfsruwmtrgaocjcyxdkovtdennrkmxwpdsxpxuarhgusizmwakrmhdwcgvfljhzcskclgrvvbrkesojyhofwqiwhiupujmkcvlywjtmbncurxxmpdskupyvvweuhbsnanzfioirecfxvmgcpwrpmbhmkdtckhvbxnsbcifhqwjjczfokovpqyjmbywtpaqcfjowxnmtirdsfeujyogbzjnjcmqyzciwjqxxgrxblvqbutqittroqadqlsdzihngpfpjovbkpeveidjpfjktavvwurqrgqdomiibfgqxwybcyovysydxyyymmiuwovnevzsjisdwgkcbsookbarezbhnwyqthcvzyodbcwjptvigcphawzxouixhbpezzirbhvomqhxkfdbokblqmrhhioyqubpyqhjrnwhjxsrodtblqxkhezubprqftrqcyrzwywqrgockioqdmzuqjkpmsyohtlcnesbgzqhkalwixfcgyeqdzhnnlzawrdgskurcxfbekbspupbduxqxjeczpmdvssikbivjhinaopbabrmvscthvoqqbkgekcgyrelxkwoawpbrcbszelnxlyikbulgmlwyffurimlfxurjsbzgddxbgqpcdsuutfiivjbyqzhprdqhahpgenjkbiukurvdwapuewrbehczrtswubthodv"))
# s = 'babad'
# for i in range(len(s)):
#     print(s[i])
