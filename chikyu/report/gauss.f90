program gauss
  implicit none
  REAL, dimension(4,4) :: a, b
  REAL, dimension(1,4) :: c
  REAL, dimension(4,1) :: mnum, x, tmp
  REAL :: ans, maxnum, temp
  INTEGER :: i, j, k(1)
  a = reshape( (/10, 1, 4, 0, 1, 10, 5, -1, 4, 5, 10, 7, 0, -1, 7, 9/), (/4, 4/) )
  c = reshape( (/15, 15, 26, 15/), (/1,4/) )  
  tmp = reshape( (/0, 0, 0, 0/), (/4,1/) )

  do i = 1, 4, 1
  ! a is matrix, 
  ! scaling
    maxnum = MAXVAL(abs(a(:, i)))
    a(:, i) = a(:, i) / maxnum
    ! list for remembering max value
    mnum(i,1) = maxnum
  end do

  do i = 1, 3, 1
    ! pivote
    k = maxloc(a(i, i:))
    ! swap row
    tmp = a(:, i)
    a(:, i) = a(:, i+k(1))
    a(:, i+k(1)) = tmp
    temp = c(1, i)
    c(1, i) = c(1, i+k(1))
    c(1, i+k(1)) = temp
    
    ! delete
    if (a(i,i) /= 0.0 .and. i /= 4) then
      do j = i + 1, 4
        if (a(i,i) /= 0.0 .and. a(i,j) /= 0.0) then
          b(i, j) = -a(i, j) / a(i, i)
          a(:,j) = a(:, j) - (a(i, j)/a(i, i))*a(:,i)
        end if
      end do
    end if
  end do

  ans = 0

  do i = 4, 1, -1
    ans = c(1,i) - SUM((/ (a(j, 2) * x(1,j) , j = i, 4) /)) / a(i,i)
    x(i,1) = ans
    ans = 0
  end do

print *, x

end program gauss

